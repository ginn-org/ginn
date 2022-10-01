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

// Lazy way to include everything with one include statement, preferably
// avoided.

#include <ginn/init/init.h>

#include <ginn/node/affine.h>
#include <ginn/node/common.h>
#include <ginn/node/conv.h>
#include <ginn/node/data.h>
#include <ginn/node/inplace.h>
#include <ginn/node/layernorm.h>
#include <ginn/node/layout.h>
#include <ginn/node/nlnode.h>
#include <ginn/node/pick.h>
#include <ginn/node/pool.h>
#include <ginn/node/prod.h>
#include <ginn/node/reduce.h>
#include <ginn/node/select.h>
#include <ginn/node/weight.h>

#include <ginn/nonlin.h>

#include <ginn/update/update.h>

#include <ginn/util/amax.h>
#include <ginn/util/cli2.h>
#include <ginn/util/csv.h>
#include <ginn/util/fmt.h>
#include <ginn/util/indexmap.h>
#include <ginn/util/lookup.h>
#include <ginn/util/parfor.h>
#include <ginn/util/sample.h>
#include <ginn/util/sto.h>
#include <ginn/util/timer.h>
#include <ginn/util/traits.h>
#include <ginn/util/tree.h>
#include <ginn/util/tty.h>
#include <ginn/util/util.h>
#include <ginn/util/wvec.h>

#include <ginn/model/lstm.h> // models
#include <ginn/model/treelstm.h>

#include <ginn/layer/common.h>
#include <ginn/layer/layer.h>
#include <ginn/layer/tree.h>
