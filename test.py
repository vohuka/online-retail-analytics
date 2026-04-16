import pandas as pd
import matplotlib.pyplot as plt

rfm = pd.read_csv('outputs/rfm_segments.csv')

# Kiểm tra xem có outlier cực đoan không
print("=== PHÂN PHỐI MONETARY ===")
print(rfm['Monetary'].describe())
print(f"\nTop 5 Monetary cao nhất:")
print(rfm['Monetary'].nlargest(5))

print(f"\nTop 5 Frequency cao nhất:")
print(rfm['Frequency'].nlargest(5))