import numpy as np


"""
Полносвязная нейронная сеть прямого распространения на numpy
Скрытые слои - ReLU, выходной слой - softmax
Обучение - обратное распространение ошибки (Adam)
"""


class Network:
    def __init__(self, sizes: list, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            self.weights.append(rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out)).astype(np.float32))
            self.biases.append(np.zeros(n_out, dtype=np.float32))

    # Прямой проход, возвращает выходы всех слоев
    def _forward(self, x: np.ndarray) -> list:
        layers = [x]
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = x @ weight + bias
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0)
            else:
                x = np.exp(x - x.max(axis=1, keepdims=True))
                x /= x.sum(axis=1, keepdims=True)
            layers.append(x)
        return layers

    # Вероятности классов для каждой строки матрицы признаков
    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._forward(np.asarray(x, dtype=np.float32))[-1]

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 20, batch: int = 64,
            lr: float = 0.003, seed: int = 1) -> None:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y)
        rng = np.random.default_rng(seed)
        params = self.weights + self.biases
        moment1 = [np.zeros_like(p) for p in params]
        moment2 = [np.zeros_like(p) for p in params]
        step = 0

        for _ in range(epochs):
            order = rng.permutation(len(x))
            for start in range(0, len(x), batch):
                rows = order[start:start + batch]
                layers = self._forward(x[rows])

                # Градиент перекрестной энтропии по выходу softmax
                delta = layers[-1].copy()
                delta[np.arange(len(rows)), y[rows]] -= 1
                delta /= len(rows)

                grads_w = [None] * len(self.weights)
                grads_b = [None] * len(self.biases)
                for i in range(len(self.weights) - 1, -1, -1):
                    grads_w[i] = layers[i].T @ delta
                    grads_b[i] = delta.sum(axis=0)
                    if i > 0:
                        delta = (delta @ self.weights[i].T) * (layers[i] > 0)

                step += 1
                for p, g, m1, m2 in zip(params, grads_w + grads_b, moment1, moment2):
                    m1 *= 0.9
                    m1 += 0.1 * g
                    m2 *= 0.999
                    m2 += 0.001 * g * g
                    p -= lr * (m1 / (1 - 0.9 ** step)) / (np.sqrt(m2 / (1 - 0.999 ** step)) + 1e-8)

    def save(self, path) -> None:
        arrays = {f"w{i}": w for i, w in enumerate(self.weights)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.biases)})
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path) -> "Network":
        data = np.load(path)
        net = cls([1, 1])
        count = len(data.files) // 2
        net.weights = [data[f"w{i}"] for i in range(count)]
        net.biases = [data[f"b{i}"] for i in range(count)]
        return net
