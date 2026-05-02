import pandas as pd
import matplotlib.pyplot as plt


baseline = pd.read_csv("baseline_history.csv")
weighted = pd.read_csv("weighted_history.csv")


# 1. validation total loss
plt.figure(figsize=(10, 5))
plt.plot(baseline["epoch"], baseline["val_total"], marker="o", label="Baseline")
plt.plot(weighted["epoch"], weighted["val_total"], marker="o", label="Weighted")
plt.xlabel("Epoch")
plt.ylabel("Validation Total Loss")
plt.title("Baseline vs Weighted: Validation Loss")
plt.legend()
plt.tight_layout()
plt.show()


# 2. parameter-wise validation loss
plt.figure(figsize=(10, 5))
plt.plot(baseline["epoch"], baseline["val_mu"], marker="o", label="Baseline mu")
plt.plot(weighted["epoch"], weighted["val_mu"], marker="o", label="Weighted mu")
plt.plot(baseline["epoch"], baseline["val_delta"], marker="o", label="Baseline delta")
plt.plot(weighted["epoch"], weighted["val_delta"], marker="o", label="Weighted delta")
plt.plot(baseline["epoch"], baseline["val_c"], marker="o", label="Baseline c")
plt.plot(weighted["epoch"], weighted["val_c"], marker="o", label="Weighted c")
plt.xlabel("Epoch")
plt.ylabel("Validation Parameter-wise Loss")
plt.title("Baseline vs Weighted: Parameter-wise Validation Loss")
plt.legend()
plt.tight_layout()
plt.show()


# 3. gradient comparison（只有 weighted 有）
plt.figure(figsize=(10, 5))
plt.plot(weighted["epoch"], weighted["g_mu"], marker="o", label="g_mu")
plt.plot(weighted["epoch"], weighted["g_delta"], marker="o", label="g_delta")
plt.plot(weighted["epoch"], weighted["g_c"], marker="o", label="g_c")
plt.xlabel("Epoch")
plt.ylabel("Mean Gradient Norm")
plt.title("Weighted Model: Gradient Norm Evolution")
plt.legend()
plt.tight_layout()
plt.show()


# 4. dynamic weights（只有 weighted 有）
plt.figure(figsize=(10, 5))
plt.plot(weighted["epoch"], weighted["w_mu"], marker="o", label="w_mu")
plt.plot(weighted["epoch"], weighted["w_delta"], marker="o", label="w_delta")
plt.plot(weighted["epoch"], weighted["w_c"], marker="o", label="w_c")
plt.xlabel("Epoch")
plt.ylabel("Mean Dynamic Weight")
plt.title("Weighted Model: Dynamic Weight Evolution")
plt.legend()
plt.tight_layout()
plt.show()


# 5. 最後一個 epoch 的比較表
final_compare = pd.DataFrame({
    "metric": [
        "val_total",
        "val_mu",
        "val_delta",
        "val_c"
    ],
    "baseline": [
        baseline["val_total"].iloc[-1],
        baseline["val_mu"].iloc[-1],
        baseline["val_delta"].iloc[-1],
        baseline["val_c"].iloc[-1],
    ],
    "weighted": [
        weighted["val_total"].iloc[-1],
        weighted["val_mu"].iloc[-1],
        weighted["val_delta"].iloc[-1],
        weighted["val_c"].iloc[-1],
    ]
})

print("\nFinal comparison table:")
print(final_compare)