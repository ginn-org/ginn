// Copyright 2022 Bloomberg Finance L.P.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ginn/dev.h>
#include <ginn/node/affine.h>
#include <ginn/node/layout.h>
#include <ginn/nonlin.h>
#include <ginn/prod.h>
#include <ginn/tensor.h>
#include <ginn/util/cli2.h>
#include <ginn/util/fmt.h>
#include <ginn/util/parfor.h>
#include <ginn/util/timer.h>
#include <iostream>

using namespace ginn;

void barrier() {
#ifdef GINN_ENABLE_GPU
  // This is needed for proper timing since cuda calls are async and cpu code
  // continues execution immediately.
  for (int i = 0; i < gpus(); i++) {
    GINN_CUDA_CALL(cudaSetDevice(i));
    GINN_CUDA_CALL(cudaDeviceSynchronize());
  }
#endif
}

void compare_gelus() {
#ifndef GINN_ENABLE_GPU
  DevPtr dev = cpu();
  size_t num_repeat = 1e2;
#else
  DevPtr dev = gpu();
  size_t num_repeat = 1e5;
#endif
  Tensor<Real> sink(dev, Shape{});

  auto gelu = GeluOp<Real>();
  auto gelu2 = Gelu2Op<Real>();

  for (Size rows : {1, 10, 100, 250, 500, 1000}) {
    for (Size cols : {100}) {
      Tensor<Real> x(dev, {rows, cols});
      x.set_random();
      Tensor<Real> y1(dev, x.shape()), y2(dev, x.shape());

      for (size_t repeat = 0; repeat < num_repeat; repeat++) {
        timer::time(std::to_string(rows) + ":Gelu",
                    [&]() { gelu.forward(y1, x); });
        timer::time(std::to_string(rows) + ":Gelu2",
                    [&]() { gelu2.forward(y2, x); });
        // just to make sure things are not optimized away by compiler
        sink = sink.view<0>() + y1.t().sum() + y2.t().sum();
      }

      Tensor<Real> dx(dev, x.shape()), dy1(dev, x.shape()), dy2(dev, x.shape());
      dy1.set_random();
      dy2.set_random();

      for (size_t repeat = 0; repeat < num_repeat; repeat++) {
        dx.set_zero();
        timer::time(std::to_string(rows) + ":GeluBackward",
                    [&]() { gelu.backward(dx, dy1, x, y1, true); });
        dx.set_zero();
        timer::time(std::to_string(rows) + ":Gelu2Backward",
                    [&]() { gelu2.backward(dx, dy2, x, y2, true); });
        // just to make sure things are not optimized away by compiler
        sink = sink.view<0>() + dx.t().sum();
      }
    }
  }

  std::cout << sink.maybe_copy_to(cpu()).v() << std::endl;

  timer::print(timer::TimerSort::Name);
  timer::reset();
}

void compare_matmuls() {
#ifdef GINN_ENABLE_GPU
  DevPtr dev = gpu();
  Tensor<Real> sink(dev, Shape{});
  sink.set_zero();
  size_t num_repeat = 1e3;

  Tensor<Real> dummy(sink);
  internal::gpu_prod(
      sink, dummy, dummy); // first prod is always slow. JIT? Cublas init?

  cudaDeviceSynchronize();

  for (Size inner : {1, 10, 100, 250, 500, 1000, 4000}) {
    for (Size outer : {1, 10, 100, 250, 500, 1000, 4000, 16000}) {
      Tensor<Real> a(dev, {outer, inner}), b(dev, {inner, outer}),
          c(dev, {outer, outer});
      a.set_random();
      b.set_random();
      for (size_t repeat = 0; repeat < num_repeat; repeat++) {
        timer::time(std::to_string(inner) + ";" + std::to_string(outer) +
                        ":CublasGemm",
                    [&]() {
                      internal::gpu_prod(c, a, b);
                      cudaDeviceSynchronize();
                    });
        sink = sink.view<0>() + c.t().sum();

        Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
            Eigen::IndexPair<int>(1, 0)};
        timer::time(std::to_string(inner) + ";" + std::to_string(outer) +
                        ":EigenContraction",
                    [&]() {
                      c = a.t().contract(b.t(), product_dims);
                      cudaDeviceSynchronize();
                    });
        sink = sink.view<0>() + c.t().sum();
      }
    }
  }
  std::cout << sink.maybe_copy_to(cpu()).v() << std::endl;

  timer::print(timer::TimerSort::Name);
  timer::reset();
#endif
}

void compare_cpu_matmuls() {
  Tensor<Real> sink(cpu(), Shape{});
  sink.set_zero();
  size_t num_repeat = 1;

  for (Size rows : {1, 10, 100, 250}) {
    auto x = rows, y = rows + 1, z = rows + 2;
    Tensor<Real> a(cpu(), {x, y}), b(cpu(), {y, z}), c(cpu(), {x, z});
    a.set_random(); // x y
    b.set_random(); // y z
    for (size_t repeat = 0; repeat < num_repeat; repeat++) {
      timer::time(std::to_string(rows) + ":Eigen MatMul",
                  [&]() { c.m() = a.m() * b.m(); });
      sink = sink.view<0>() + c.t().sum();

      timer::time(std::to_string(rows) + ":Hacky Broadcast", [&]() {
        c = (a.t().reshape(Index<3>{x, y, 1}).broadcast(Index<3>{1, 1, z}) *
             b.t().reshape(Index<3>{1, y, z}).broadcast(Index<3>{x, 1, 1}))
                .sum(Index<1>{1});
      });
      sink = sink.view<0>() + c.t().sum();
    }
  }
  std::cout << sink.maybe_copy_to(cpu()).v() << std::endl;

  timer::print(timer::TimerSort::Name);
  timer::reset();
}

/*
void compare_batched_matmuls() {
#ifdef GINN_ENABLE_GPU
  DevPtr dev = gpu();
  Tensor<Real> sink(dev, Shape{});
  sink.set_zero();
  size_t num_repeat = 1e3;

  Tensor<Real> dummy(sink);
  gpu_prod(sink, dummy, dummy); // first prod is always slow. JIT? Cublas init?

  Tensor<Real> a(dev, {2, 3, 4}), b(dev, {3, 5, 4}), c(dev, {2, 5, 4}),
      c2(dev, {2, 5, 4}), c3(dev, {2, 5, 4});
  Tensor<Real> at(dev, {3, 2, 4}), bt(dev, {5, 3, 4});
  a.set_random();
  at.set_random();
  b.set_random();
  bt.set_random();
  gpu_batched_prod(c, a, b);
  gpu_batched_add_prod_t1(c2, at, b);
  gpu_batched_add_prod_t2(c3, a, bt);
  sink = sink.view<0>() + c.t().sum();
  std::cout << sink.maybe_copy_to(cpu()).v() << std::endl;

  a.move_to(cpu());
  b.move_to(cpu());
  c.move_to(cpu());
  c2.move_to(cpu());
  Tensor<Real> c_(c);
  for (size_t i = 0; i < 4; i++) {
    Tensor<Real> a_i, b_i, c_i;
    a_i.map(a, {2, 3}, i * 2 * 3);
    b_i.map(b, {3, 5}, i * 3 * 5);
    c_i.map(c, {2, 5}, i * 2 * 5);
    c_i.m() = a_i.m() * b_i.m();
  }

  std::cout << c.v().transpose() << std::endl;
  std::cout << c_.v().transpose() << std::endl;

  // timer::print(timer::TimerSort::Name);
  // timer::reset();
#endif
}*/

void affine() {
#ifdef GINN_ENABLE_GPU
  DevPtr dev = gpu();
  Tensor<Real> sink(dev, Shape{});
  sink.set_zero();
  Tensor<Real> dummy(sink);
  internal::gpu_prod(
      sink, dummy, dummy); // first prod is always slow. JIT? Cublas init?
  size_t num_repeat = 1e3;
#else
  DevPtr dev = cpu();
  size_t num_repeat = 3;
#endif
  barrier();

  for (Size hdim : {1, 10, 100, 250, 500, 1000, 4000}) {
    for (Size batches : {1, 10, 100, 250, 500, 1000, 4000, 16000}) {
      for (size_t i = 0; i < num_repeat; i++) {
        auto x = Data<Real>(dev, Shape{hdim, batches});
        auto W = Data<Real>(dev, Shape{hdim, hdim});
        auto b = Data<Real>(dev, Shape{hdim});
        for (auto n : {x, W, b}) { n->value().set_random(); }
        auto y = Affine(W, x, b);
        Graph g(y);

        std::string prefix =
            std::to_string(hdim) + ";" + std::to_string(batches);

        timer::time(prefix + " fwd", [&]() {
          g.forward();
          barrier();
        });
        timer::time(prefix + " rg", [&]() {
          g.reset_grad();
          barrier();
        });
        timer::time(prefix + " bwd", [&]() {
          g.backward(1.);
          barrier();
        });
      }
    }
  }

  timer::print(timer::TimerSort::Name);
  timer::reset();
}

void multi_gpu() {
#ifdef GINN_ENABLE_GPU
  GINN_ASSERT(gpus() >= 2, "multi_gpu benchmark requires at least two gpus!");
  Size batches = 256;
  size_t num_repeats = 1e3;

  std::vector<Tensor<Real>> sink;
  std::vector<PreallocGpu> pgpu;

  size_t max_hdim = 4096;
  size_t mem = max_hdim * max_hdim * 2;
  pgpu.emplace_back(0, mem);
  pgpu.emplace_back(1, mem);
  pgpu.emplace_back(1, mem);
  for (bool prealloc : {true, false}) {
    for (Size hdim : {512, 1024, 2048, 4096}) {
      std::vector<NodePtr<Real>> x, W, b;

      for (int i = 0; i < 3; i++) {
        int j = std::min(1, i);
        DevPtr dev = gpu(j);
        sink.emplace_back(dev, Shape{});
        x.push_back(Random(dev, Shape{hdim, batches}));
        W.push_back(Random(dev, Shape{hdim, hdim}));
        b.push_back(Random(dev, Shape{hdim}));
      }
      barrier();

      auto work = [&](int j, int /*thread id*/) {
        for (size_t i = 0; i < num_repeats; i++) {
          NodePtr<Real> x_ = x[j];
          if (prealloc) { x_ = DeviceView(x_, pgpu[j]); }
          auto y = Affine(W[j], x_, b[j]);
          Graph(y).forward();
          sink[j] = sink[j].view<0>() + y->value().t().sum();
          pgpu[j].reset();
        }
      };

      timer::time(fmt::format("{} {:>4} different gpus", prealloc, hdim),
                  [&]() {
                    parallel_for(0, 2, work, /*threads*/ 1);
                    barrier();
                  });

      timer::time(fmt::format("{} {:>4} same gpu", prealloc, hdim), [&]() {
        parallel_for(1, 3, work, /*threads*/ 2);
        barrier();
      });
    }
  }

  std::cout << sink[0].item() + sink[1].item() + sink[2].item() << std::endl;

  timer::print(timer::TimerSort::Name);
  timer::reset();
#endif
}

int main(int argc, char** argv) {
  using namespace ginn;

  std::unordered_map<std::string, void (*)()> benchmarks{
      {"cpu-matmul", compare_cpu_matmuls},
      {"gelu", compare_gelus},
      {"affine", affine},
      {"matmul", compare_matmuls},
      {"multi-gpu", multi_gpu},
  };

  std::vector<std::string> keys;
  for (auto& p : benchmarks) { keys.push_back(p.first); }

  std::vector<std::string> names;

  Args args;
  args(Arg(names)
           .meta("benchmark name")
           .help("name(s) of the benchmarks to run")
           .required()
           .require_choices(keys));
  args.parse(argc, argv);

  for (const auto& name : names) {
    std::cout << "Running benchmark: " << name << "..." << std::endl;
    benchmarks[name]();
  }

  return 0;
}
