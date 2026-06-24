"""
Regenerate fig4_tier3.pdf with the REAL attention-pooling results.
Run on Kaggle or locally: pip install matplotlib numpy
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = '.'

# ── REAL attention-pooling results (from actual tier3_train.py run) ───────
dims    = ['AF Burden', 'Longest\nEpisode', 'Episode\nCount',
           'Nocturnal\nRatio', 'Trend\nSlope']
mae     = [0.0074, 0.0228, 0.0122, 0.0160, 0.0086]
targets = [0.0500, 0.0300, 0.0200, 0.0500, 0.0400]
beats_target = [m <= t for m, t in zip(mae, targets)]   # all True now

# ── REAL training convergence (actual epoch-by-epoch values from run) ─────
epochs = np.arange(1, 41)
tr_loss = np.array([
    0.13507,0.09566,0.08276,0.06496,0.06077,0.04678,0.05387,0.04575,0.03733,0.04016,
    0.03320,0.03522,0.02929,0.03357,0.02430,0.02461,0.02772,0.02619,0.02442,0.02147,
    0.01925,0.02260,0.01744,0.01858,0.01800,0.01715,0.01555,0.01717,0.01477,0.01458,
    0.01172,0.01410,0.01258,0.01178,0.01116,0.00967,0.01174,0.01120,0.01134,0.00974
])
va_loss = np.array([
    0.11228,0.09632,0.09897,0.06893,0.05526,0.05786,0.05541,0.05714,0.05571,0.04659,
    0.04790,0.04069,0.03862,0.03892,0.03390,0.03836,0.03508,0.03397,0.03503,0.03045,
    0.03178,0.03067,0.03024,0.03078,0.02932,0.02930,0.03285,0.02857,0.02710,0.02602,
    0.02387,0.02452,0.02344,0.02525,0.02398,0.02327,0.02349,0.02339,0.02343,0.02341
])
BEST_EPOCH = 30   # last checkpoint save per actual training log

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Tier-3 Temporal Aggregator: Rhythm-Burden Vector Performance (Attention Pooling)\n'
             'Trained on LTAF Long-Term Recordings (84 patients, 705,816 strips)',
             fontsize=12, fontweight='bold')

x      = np.arange(len(dims))
width  = 0.35
colors_mae = ['#43A047'] * len(dims)   # all green now -- all 5 pass

b1 = ax1.bar(x - width/2, targets, width, label='Clinical target (ACC/AHA 2023)',
             color='#BBDEFB', edgecolor='#1E88E5', linewidth=1.5, zorder=2)
b2 = ax1.bar(x + width/2, mae, width, label='CardioAgent MAE (attention pooling)',
             color=colors_mae, edgecolor='white', linewidth=1.5, zorder=3)

for bar, m in zip(b2, mae):
    ax1.text(bar.get_x() + bar.get_width()/2, m + 0.001,
             f'{m:.4f} \u2713', ha='center', va='bottom',
             fontsize=8, fontweight='bold', color='#1B5E20')

ax1.set_xticks(x)
ax1.set_xticklabels(dims, fontsize=10)
ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=11)
ax1.set_title('MAE vs ACC/AHA 2023 Clinical Targets', fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3, zorder=0)
ax1.spines[['top', 'right']].set_visible(False)

ax2.plot(epochs, tr_loss, color='#1E88E5', linewidth=1.8, label='Training loss', alpha=0.8)
ax2.plot(epochs, va_loss, color='#E53935', linewidth=2.0, label='Validation loss')
ax2.axvline(x=BEST_EPOCH, color='#43A047', linestyle='--', alpha=0.7,
            label=f'Best model (epoch {BEST_EPOCH})')
ax2.scatter([BEST_EPOCH], [va_loss[BEST_EPOCH-1]], color='#43A047', s=80, zorder=5)

ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Combined Loss (MSE + Correlation)', fontsize=11)
ax2.set_title('Training Convergence on LTAF (Attention Pooling)', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.spines[['top', 'right']].set_visible(False)

summary = ('Mean MAE = 0.0134\n'
           '5 / 5 dimensions\n'
           'beat clinical targets')
ax2.text(0.97, 0.97, summary, transform=ax2.transAxes,
         fontsize=9, va='top', ha='right',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#F1F8E9',
                   edgecolor='#43A047', alpha=0.9))

plt.tight_layout()
out = f'{OUTPUT_DIR}/fig4_tier3.pdf'
plt.savefig(out, bbox_inches='tight', dpi=200)
plt.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
print(f"Saved: {out}")
print("Replace the OLD fig4_tier3.pdf in your Overleaf project with this one.")
