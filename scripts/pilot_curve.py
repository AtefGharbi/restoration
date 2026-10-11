import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_experiments import context, evaluate
from restoration.runner import train
d = json.load(open('instances/ieee33.json'))
ctx = context('ieee33')
for n in (int(a) for a in sys.argv[1:]):
    t = time.time(); lr, curve, tm = train(ctx, 'B7', 123, d['train'], n)
    # NOTE: budget calibration must not touch the test set; the first version of this script printed
    # test ENS for B7 at 400/1600 episodes (disclosed in the response letter).  Training return only.
    ens = float('nan')
    print(n, 'train %.0fs' % (time.time() - t), 'ret first/last100 %.3f %.3f' % (np.mean(curve[:100]), np.mean(curve[-100:])),
          {k: round(v, 1) for k, v in tm.items()}, flush=True)
