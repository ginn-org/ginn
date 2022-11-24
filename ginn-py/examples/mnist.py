# Copyright 2022 Bloomberg Finance L.P.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple
import time

import ginn
import numpy as np
import typer


def mnist_reader(
    fname: str, bs: int, dev: ginn.Device
) -> Tuple[List[ginn.RealDataNode], List[ginn.IntDataNode]]:
    x = np.loadtxt(fname, delimiter=",")
    y = np.transpose(x[:, 0:1])
    x = np.transpose(x[:, 1:])
    x *= 1.0 / 255.0

    indices = np.arange(bs, x.shape[1], bs)
    xs = np.array_split(x, indices, axis=1)
    ys = np.array_split(y, indices, axis=1)
    xs = [ginn.Data(x) for x in xs]
    ys = [ginn.Data(y).int() for y in ys]

    for x in xs:
        x.move_to(dev)
    for y in ys:
        y.move_to(dev)

    return xs, ys


def main(
    train_file: str = typer.Option(...),
    test_file: str = typer.Option(...),
    dimx: int = 784,
    dimy: int = 10,
    dimh: int = 512,
    lr: float = 1e-3,
    bs: int = 64,
    epochs: int = 10,
    # seed: int = 123457,
    gpu: bool = False,
):
    dev = ginn.gpu() if gpu else ginn.cpu()

    X, Y = mnist_reader(train_file, bs, dev)
    Xt, Yt = mnist_reader(test_file, bs, dev)

    W = ginn.Weight(dev, [dimh, dimx])
    b = ginn.Weight(dev, [dimh])
    U = ginn.Weight(dev, [dimy, dimh])
    c = ginn.Weight(dev, [dimy])

    weights = [W, b, U, c]
    ginn.Uniform().init(weights)
    updater = ginn.Adam(lr)

    # Single instance (batch)
    def run_instance(x, y, train: bool):
        h = ginn.Sigmoid(ginn.Affine([W, x, b]))
        y_ = ginn.Affine([U, h, c])
        loss = ginn.Sum(ginn.PickNegLogSoftmax(y_, y))

        g = ginn.Graph(loss)
        g.forward()

        ycpu = y.value.maybe_copy_to(ginn.cpu())
        y_cpu = y_.value.maybe_copy_to(ginn.cpu())
        correct = np.sum(np.argmax(y_cpu.m(), 0) == ycpu.v())

        if train:
            g.reset_grad()
            g.backward(1.0)
            updater.update(weights)

        return correct, y.size

    # Single epoch
    def run_epoch(X_, Y_, train: bool):
        perm = np.random.permutation(len(X_)) if train else range(len(X_))
        correct, total = 0, 0
        for j in perm:
            c, t = run_instance(X_[j], Y_[j], train)
            correct += c
            total += t
        return correct / total

    # Main training loop
    print("TrErr%\tTstErr%\tsecs")
    for e in range(epochs):
        tic = time.time()
        acc = run_epoch(X, Y, train=True)
        acct = run_epoch(Xt, Yt, train=False)
        toc = time.time() - tic
        print(f"{100*(1-acc):6.3f}\t{100*(1-acct):6.3f}\t{toc:4.2f}")


if __name__ == "__main__":
    typer.run(main)
