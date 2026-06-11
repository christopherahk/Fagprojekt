import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

df = pd.read_csv("training_accuracy.csv")

sidste_epoke = df[df['epoch'] == 9].copy()


sidste_epoke['label'] = pd.to_numeric(sidste_epoke['label'], errors='coerce')
sidste_epoke['pred'] = pd.to_numeric(sidste_epoke['pred'], errors='coerce')


sidste_epoke = sidste_epoke.dropna(subset=['label', 'pred'])

y_true = sidste_epoke['label'].astype(int)
y_pred = sidste_epoke['pred'].astype(int)


mål_klasser = [0, 1, 2]
klasse_navne = ["dorsi", "plantar", "none"]


print("--- KLASSIFIKATIONSRAPPORT ---")
print(classification_report(y_true, y_pred, labels=mål_klasser, target_names=klasse_navne))


cm = confusion_matrix(y_true, y_pred, labels=mål_klasser)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=klasse_navne)
disp.plot(cmap=plt.cm.Blues)

plt.title("Confusion Matrix - Sidste Epoke (Supervised)")
plt.show()
