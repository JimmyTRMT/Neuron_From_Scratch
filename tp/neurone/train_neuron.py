import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.neuronnes import Neuron

graine = np.random.randint(0, 1000000)
np.random.seed(graine)
print(f"graine : {graine}")

X = np.vstack([np.random.normal(-2, 1.8, (100, 2)), np.random.normal(2, 1.8, (100, 2))])
y = np.r_[np.zeros(100), np.ones(100)]


def matriceConfusion(yPred, yVrai):
    vraisPositifs = np.sum((yPred == 1) & (yVrai == 1))
    vraisNegatifs = np.sum((yPred == 0) & (yVrai == 0))
    fauxPositifs = np.sum((yPred == 1) & (yVrai == 0))
    fauxNegatifs = np.sum((yPred == 0) & (yVrai == 1))
    return vraisPositifs, vraisNegatifs, fauxPositifs, fauxNegatifs


def exactitude(yPred, yVrai):
    vp, vn, fp, fn = matriceConfusion(yPred, yVrai)
    return (vp + vn) / (vp + vn + fp + fn)


neurone = Neuron()
tauxApprentissage = 0.1
epoques = 500
pertes = []

for epoque in range(epoques):
    a = neurone.forward(X)
    pertes.append(neurone.loss(a, y))
    if epoque % 50 == 0:
        # meme poids pour la perte et l'exactitude
        exact = exactitude((a >= 0.5).astype(int), y)
        print(f"epoque {epoque} : loss = {pertes[-1]:.4f}, exactitude = {exact:.4f}")
    dw, db = neurone.gradients(X, y, a)
    neurone.update(dw, db, tauxApprentissage)

a = neurone.forward(X)
yPred = neurone.predict(X)
vp, vn, fp, fn = matriceConfusion(yPred, y)

print(f"loss finale : {neurone.loss(a, y):.4f}")
print(f"vrais positifs : {vp}, vrais negatifs : {vn}, faux positifs : {fp}, faux negatifs : {fn}")
print(f"exactitude : {exactitude(yPred, y):.4f}")

dossier = Path(__file__).resolve().parents[2] / "images"
dossier.mkdir(exist_ok=True)

# courbe de perte
plt.figure()
plt.plot(pertes)
plt.xlabel("epoque")
plt.ylabel("perte")
plt.grid(True)
plt.savefig(dossier / "loss_neuron.png")
plt.close()

vraiNegatif = (yPred == 0) & (y == 0)
vraiPositif = (yPred == 1) & (y == 1)
fauxPositif = (yPred == 1) & (y == 0)
fauxNegatif = (yPred == 0) & (y == 1)

plt.figure()
plt.scatter(X[vraiNegatif, 0], X[vraiNegatif, 1], color="tab:blue", label="vrai negatif")
plt.scatter(X[vraiPositif, 0], X[vraiPositif, 1], color="tab:orange", label="vrai positif")
plt.scatter(X[fauxPositif, 0], X[fauxPositif, 1], color="green", label="faux positif")
plt.scatter(X[fauxNegatif, 0], X[fauxNegatif, 1], color="red", label="faux negatif")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.savefig(dossier / "predictions_neuron.png")
plt.close()
