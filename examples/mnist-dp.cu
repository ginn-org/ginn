#include <iostream>

#include <ginn/node/affine.h>
#include <ginn/node/common.h>
#include <ginn/node/layout.h>
#include <ginn/node/nlnode.h>
#include <ginn/node/pick.h>
#include <ginn/node/reduce.h>
#include <ginn/node/weight.h>

#include <ginn/init/init.h>
#include <ginn/update/update.h>

#include <ginn/util/amax.h>
#include <ginn/util/cli2.h>
#include <ginn/util/csv.h>
#include <ginn/util/fmt.h>
#include <ginn/util/parfor.h>
#include <ginn/util/timer.h>
#include <ginn/util/util.h>

#include <ginn/metric.h>

using namespace ginn;

using Indices = std::vector<Int>;

std::tuple<std::vector<DataPtr<Real>>, std::vector<Indices>>
mnist_reader(const std::string& fname, Size bs = 64) {
  Matrix<Real> X, Y, tmp;
  tmp = read_csv<Matrix<Real>>(fname, ',');

  X = tmp.rightCols(tmp.cols() - 1).transpose();
  X *= (1. / 255);
  Y = tmp.leftCols(1).transpose();

  Size n = X.cols();
  Size d = X.rows();

  std::vector<DataPtr<Real>> Xs;
  std::vector<Indices> Ys;

  for (Size i = 0; i < n; i += bs) {
    Size batch_size = std::min(bs, n - i);
    auto x = FixedData(cpu(), {d, batch_size});
    x->value().m() = X.middleCols(i, batch_size);
    // x->move_to(dev());
    Xs.push_back(x);
    Indices y(batch_size);
    for (Size j = 0; j < batch_size; j++) { y[j] = Y(i + j); }
    Ys.push_back(y);
  }

  return {Xs, Ys};
}

int main(int argc, char** argv) {
  std::string train_file, test_file;
  Size dimx = 784;
  Size dimy = 10;
  Size dimh = 512;
  Real lr = 1e-3;
  Size bs = 64;
  size_t epochs = 10;
  size_t seed = 123457;

  Args args;
  args.add(Arg(train_file)
               .name("train-file")
               .meta("path")
               .help("path to training set in csv format")
               .required());
  args.add(Arg(test_file)
               .name("test-file")
               .meta("path")
               .help("path to testing set in csv format")
               .required());
  args.add(Arg(dimh).name("d,hidden-dim").help("hidden dimension"));
  args.add(Arg(dimx).name("input-dim").help("input dimension"));
  args.add(Arg(dimy).name("output-dim").help("output dimension"));
  args.add(Arg(lr)
               .name("l,learning-rate")
               .help("learning rate")
               .suggest_range(1e-5, 1e-3)
               .require_range(0, 1));
  args.add(Arg(epochs).name("e,epochs").help("number of epochs to train"));
  args.add(Arg(bs).name("b,batch-size").help("batch size"));
  args.add(Arg(seed).name("s,seed").help("random seed"));
  args.parse(argc, argv);

  srand(seed);

  // TODO: is this really needed?
  GINN_CUDA_CALL(cudaSetDevice(1));
  GINN_CUDA_CALL(cudaStreamSynchronize(gpu(0)->stream()));
  GINN_CUDA_CALL(cudaSetDevice(0));
  GINN_CUDA_CALL(cudaStreamSynchronize(gpu(1)->stream()));

  auto [X, Y] = mnist_reader(train_file, bs);
  auto [Xt, Yt] = mnist_reader(test_file, bs);

  auto W0 = Weight(gpu(0), {dimh, dimx});
  auto b0 = Weight(gpu(0), {dimh});
  auto U0 = Weight(gpu(0), {dimy, dimh});
  auto c0 = Weight(gpu(0), {dimy});
  // create tied copies in the same gpu (0) with shared value but unshared grads
  auto W1 = W0->copy(Copy::Tied);
  auto b1 = b0->copy(Copy::Tied);
  auto U1 = U0->copy(Copy::Tied);
  auto c1 = c0->copy(Copy::Tied);

  std::vector<WeightPtr<Real>> weights = {W0, b0, U0, c0};
  ginn::init::Uniform<Real>().init(weights);

  ginn::update::Adam<Real> updater(lr);

  // Single instance (batch)
  auto pass = [&](const DataPtr<Real>& x,
                  Indices& y,
                  auto& acc,
                  unsigned rank,
                  bool train) {
    NodePtr<Real> W_, b_, U_, c_;
    if (rank == 1) {
      W_ = DeviceTransfer(W1, gpu(rank));
      b_ = DeviceTransfer(b1, gpu(rank));
      U_ = DeviceTransfer(U1, gpu(rank));
      c_ = DeviceTransfer(c1, gpu(rank));
    } else if (rank == 0) {
      W_ = W0;
      b_ = b0;
      U_ = U0;
      c_ = c0;
    } else {
      GINN_THROW("TODO");
    }
    auto x_ = DeviceTransfer(x, gpu(rank));

    auto h = Affine<SigmoidOp>(W_, x_, b_);
    auto y_ = Affine(U_, h, c_);
    auto loss = Sum(PickNegLogSoftmax(y_, y));

    auto graph = Graph(loss);
    graph.forward();
    acc.batched_add(argmax(y_->value(), 0).maybe_copy_to(cpu()), y);

    if (train) {
      graph.reset_grad();
      graph.backward(1.);
      // updater.update(weights);
    }
  };

  std::mt19937 g(seed);

  // Single epoch
  auto pass_data = [&](auto& X, auto& Y, bool train) {
    metric::Accuracy<Int> acc;
    auto perm = train ? randperm(X.size(), g) : iota(X.size());
    for (size_t i = 0; i < X.size(); i += 2) {
      parallel_for(
          0,
          2,
          [&](size_t rank, size_t) {
            if (i + rank >= perm.size()) { return; }
            size_t j = perm[i + rank];
            if (j >= X.size()) { return; }
            pass(X[j], Y[j], acc, rank, train);
          },
          2);
      if (train) { updater.update({W0, b0, U0, c0, W1, b1, U1, c1}); }
    }
    return 100. * (1. - acc.eval());
  };

  // Main training loop
  std::cout << "TrErr%\tTstErr%\tsecs" << std::endl;
  for (size_t e = 0; e < epochs; e++) {
    using namespace ginn::literals;
    timer::tic();
    std::cout << ("{:6.3f}\t"_f, pass_data(X, Y, true)) << std::flush;
    std::cout << ("{:6.3f}\t"_f, pass_data(Xt, Yt, false)) << timer::toc() / 1e6
              << std::endl;
  }

  return 0;
}
