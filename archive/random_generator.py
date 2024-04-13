import numpy as np

def generate_surprisal_values(text, threshold=0.5):
    words = text.split()
    surprisal_values = np.random.rand(len(words))
    surprisal_values = np.insert(surprisal_values, 0, 0)
    surprisal_values = np.convolve(surprisal_values, np.array([0.5, 0.5]), mode = "valid")
    classification = "fake news" if np.mean(surprisal_values) > threshold else "true news"
    return classification, surprisal_values, words

if __name__ == "__main__":
    test_text = "On a rainy Tuesday afternoon"
    classification, surprisal_values, words = generate_surprisal_values(test_text)
    print(f"Classification: {classification}")
    print(f"Surprisal Values: {surprisal_values}")
    print(f"Words: {words}")