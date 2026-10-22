import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;

public class FriendlyLLM implements Serializable {
    private static final long serialVersionUID = 1L;

    // ===== Hyperparameters =====
    private static final int EMBEDDING_DIMENSION = 96; // increased complexity
    private static final int HIDDEN_STATE_SIZE = 192; // more expressive
    private static final int SEQUENCE_LENGTH = 50;
    private static final double LEARNING_RATE = 0.005;
    private static final double GRAD_CLIP = 5.0;
    private static final int SAMPLE_LENGTH = 50;
    private static final long RANDOM_SEED = 42L;

    // ===== Model parameters =====
    private double[][] embeddingMatrix;
    private double[][] inputToHiddenWeights;
    private double[][] hiddenToHiddenWeights;
    private double[] hiddenBias;
    private double[][] hiddenToOutputWeights;
    private double[] outputBias;

    // ===== Data =====
    private String trainingText;
    private List<Character> vocabularyList;
    private Map<Character, Integer> charToIndex;
    private Map<Integer, Character> indexToChar;
    private int vocabularySize;

    private transient Random random;
    private transient ForkJoinPool threadPool;

    public FriendlyLLM() {
        random = new Random(RANDOM_SEED);
        threadPool = new ForkJoinPool(); // use all CPU cores
    }

    public static void main(String[] args) throws Exception {
        FriendlyLLM model = new FriendlyLLM();
        model.loadOrCreateData("data.txt");

        boolean loaded = model.loadModel("model.dat");
        if (!loaded) {
            System.out.println("No existing model found. Initializing new model.");
            model.initializeModelParameters();
        } else {
            System.out.println("Loaded model from file. Continuing training.");
        }

        Scanner scanner = new Scanner(System.in);
        boolean run = true;
        while (run) {
            System.out.println("Action? (y = train, m = message, i = info, r = reset, q = quit, p=prompt)");
            String input = scanner.nextLine().trim();
            switch (input) {
                case "y" -> model.continuousTrain();
                case "m" -> System.out.println("Sample message:\n" + model.sampleMessage());
                case "i" -> model.printInfo();
                case "r" -> model.resetModel("model.dat");
                case "q" -> {
                    model.saveModel("model.dat");
                    run = false;
                }
                case "p" -> {
                    System.out.print("Prompt: ");
                    String prompt = scanner.nextLine();
                    System.out.println("AI:");
                    System.out.println(model.respondToPrompt(prompt, 200));
                }

                default -> System.out.println("Unknown command.");
            }
        }
    }

    // ===== Data Loading =====
    private void loadOrCreateData(String fileName) throws IOException {
        Path path = Paths.get(fileName);
        if (!Files.exists(path)) throw new FileNotFoundException(fileName + " not found.");
        byte[] bytes = Files.readAllBytes(path);
        trainingText = new String(bytes).replace("\r", "");
        buildVocabulary();
    }

    private void buildVocabulary() {
        Set<Character> vocabSet = new LinkedHashSet<>();
        for (char c : trainingText.toCharArray()) vocabSet.add(c);
        vocabularyList = new ArrayList<>(vocabSet);
        vocabularySize = vocabularyList.size();
        charToIndex = new HashMap<>();
        indexToChar = new HashMap<>();
        for (int i = 0; i < vocabularyList.size(); i++) {
            charToIndex.put(vocabularyList.get(i), i);
            indexToChar.put(i, vocabularyList.get(i));
        }
        System.out.println("Vocabulary size: " + vocabularySize);
    }

    // ===== Model Initialization =====
    private void initializeModelParameters() {
        embeddingMatrix = randomMatrix(vocabularySize, EMBEDDING_DIMENSION, 0.01);
        inputToHiddenWeights = randomMatrix(HIDDEN_STATE_SIZE, EMBEDDING_DIMENSION, 0.1);
        hiddenToHiddenWeights = randomMatrix(HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE, 0.1);
        hiddenBias = new double[HIDDEN_STATE_SIZE];
        hiddenToOutputWeights = randomMatrix(vocabularySize, HIDDEN_STATE_SIZE, 0.1);
        outputBias = new double[vocabularySize];
    }

    // ===== Training =====
    private void continuousTrain() {
        System.out.println("Training... press ENTER to stop.");
        try {
            while (System.in.available() == 0) {
                trainEpoch();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Training stopped by user.");
    }

    private void trainEpoch() {
        int textLength = trainingText.length();
        int pointer = 0;
        int steps = 0;

        List<Callable<Void>> tasks = new ArrayList<>();

        while (pointer + SEQUENCE_LENGTH + 1 <= textLength) {
            int start = pointer;
            tasks.add(() -> {
                trainSequence(start);
                return null;
            });
            pointer += SEQUENCE_LENGTH;
        }
        threadPool.invokeAll(tasks);
    }

    private void trainSequence(int start) {
        int[] inputIndices = new int[SEQUENCE_LENGTH];
        int[] targetIndices = new int[SEQUENCE_LENGTH];

        for (int t = 0; t < SEQUENCE_LENGTH; t++) {
            char inputChar = trainingText.charAt(start + t);
            char targetChar = trainingText.charAt(start + t + 1);
            Integer inIdx = charToIndex.get(inputChar);
            Integer tarIdx = charToIndex.get(targetChar);
            if (inIdx == null || tarIdx == null) return; // skip unknown
            inputIndices[t] = inIdx;
            targetIndices[t] = tarIdx;
        }

        ForwardCache cache = forwardPass(inputIndices);
        Gradients grad = backwardPass(cache, inputIndices, targetIndices);
        clipGradients(grad);
        applyGradients(grad, LEARNING_RATE);
    }

    // ===== Forward / Backward =====
    private ForwardCache forwardPass(int[] inputIndices) {
        ForwardCache cache = new ForwardCache(SEQUENCE_LENGTH, HIDDEN_STATE_SIZE, vocabularySize);
        double[] prevHidden = new double[HIDDEN_STATE_SIZE];
        for (int t = 0; t < SEQUENCE_LENGTH; t++) {
            int idx = inputIndices[t];
            double[] x = embeddingMatrix[idx];
            double[] h = new double[HIDDEN_STATE_SIZE];
            for (int i = 0; i < HIDDEN_STATE_SIZE; i++) {
                double sum = hiddenBias[i];
                for (int j = 0; j < EMBEDDING_DIMENSION; j++) sum += inputToHiddenWeights[i][j] * x[j];
                for (int j = 0; j < HIDDEN_STATE_SIZE; j++) sum += hiddenToHiddenWeights[i][j] * prevHidden[j];
                h[i] = Math.tanh(sum);
            }
            cache.hiddenStates[t] = h;
            prevHidden = h;

            double[] logits = new double[vocabularySize];
            for (int i = 0; i < vocabularySize; i++) {
                double sum = outputBias[i];
                for (int j = 0; j < HIDDEN_STATE_SIZE; j++) sum += hiddenToOutputWeights[i][j] * h[j];
                logits[i] = sum;
            }
            cache.probabilities[t] = softmax(logits);
        }
        return cache;
    }

    private Gradients backwardPass(ForwardCache cache, int[] inputIndices, int[] targetIndices) {
        Gradients grad = new Gradients();
        double[] dhNext = new double[HIDDEN_STATE_SIZE];

        for (int t = SEQUENCE_LENGTH - 1; t >= 0; t--) {
            double[] probs = Arrays.copyOf(cache.probabilities[t], vocabularySize);
            probs[targetIndices[t]] -= 1.0;

            for (int i = 0; i < vocabularySize; i++) {
                for (int j = 0; j < HIDDEN_STATE_SIZE; j++)
                    grad.dHiddenToOutput[i][j] += probs[i] * cache.hiddenStates[t][j];
                grad.dOutputBias[i] += probs[i];
            }

            double[] dh = new double[HIDDEN_STATE_SIZE];
            for (int i = 0; i < HIDDEN_STATE_SIZE; i++) {
                for (int j = 0; j < vocabularySize; j++) dh[i] += hiddenToOutputWeights[j][i] * probs[j];
                dh[i] += dhNext[i];
                dh[i] *= (1 - cache.hiddenStates[t][i] * cache.hiddenStates[t][i]);
            }

            for (int i = 0; i < HIDDEN_STATE_SIZE; i++) {
                for (int j = 0; j < EMBEDDING_DIMENSION; j++)
                    grad.dInputToHidden[i][j] += dh[i] * embeddingMatrix[inputIndices[t]][j];
                grad.dHiddenBias[i] += dh[i];
            }

            for (int i = 0; i < HIDDEN_STATE_SIZE; i++)
                for (int j = 0; j < HIDDEN_STATE_SIZE; j++)
                    grad.dHiddenToHidden[i][j] += dh[i] * (t > 0 ? cache.hiddenStates[t - 1][j] : 0);

            for (int i = 0; i < EMBEDDING_DIMENSION; i++)
                grad.dEmbedding[inputIndices[t]][i] += dh[i % HIDDEN_STATE_SIZE] * inputToHiddenWeights[i % HIDDEN_STATE_SIZE][i];

            dhNext = new double[HIDDEN_STATE_SIZE];
            for (int i = 0; i < HIDDEN_STATE_SIZE; i++)
                for (int j = 0; j < HIDDEN_STATE_SIZE; j++)
                    dhNext[j] += dh[i] * hiddenToHiddenWeights[i][j];
        }

        return grad;
    }

    private void clipGradients(Gradients grad) {
        for (double[][] mat : new double[][][]{grad.dEmbedding, grad.dInputToHidden, grad.dHiddenToHidden, grad.dHiddenToOutput})
            for (int i = 0; i < mat.length; i++)
                for (int j = 0; j < mat[i].length; j++)
                    mat[i][j] = Math.max(Math.min(mat[i][j], GRAD_CLIP), -GRAD_CLIP);
        for (double[] arr : new double[][]{grad.dHiddenBias, grad.dOutputBias})
            for (int i = 0; i < arr.length; i++)
                arr[i] = Math.max(Math.min(arr[i], GRAD_CLIP), -GRAD_CLIP);
    }

    private void applyGradients(Gradients grad, double lr) {
        for (int i = 0; i < embeddingMatrix.length; i++)
            for (int j = 0; j < embeddingMatrix[i].length; j++)
                embeddingMatrix[i][j] -= lr * grad.dEmbedding[i][j];

        for (int i = 0; i < inputToHiddenWeights.length; i++)
            for (int j = 0; j < inputToHiddenWeights[i].length; j++)
                inputToHiddenWeights[i][j] -= lr * grad.dInputToHidden[i][j];

        for (int i = 0; i < hiddenToHiddenWeights.length; i++)
            for (int j = 0; j < hiddenToHiddenWeights[i].length; j++)
                hiddenToHiddenWeights[i][j] -= lr * grad.dHiddenToHidden[i][j];

        for (int i = 0; i < hiddenBias.length; i++)
            hiddenBias[i] -= lr * grad.dHiddenBias[i];

        for (int i = 0; i < hiddenToOutputWeights.length; i++)
            for (int j = 0; j < hiddenToOutputWeights[i].length; j++)
                hiddenToOutputWeights[i][j] -= lr * grad.dHiddenToOutput[i][j];

        for (int i = 0; i < outputBias.length; i++)
            outputBias[i] -= lr * grad.dOutputBias[i];
    }

    // ===== Sampling =====
    private String sampleMessage() {
        int seedIdx = random.nextInt(vocabularySize);
        StringBuilder sb = new StringBuilder();
        sb.append(indexToChar.get(seedIdx));
        double[] hidden = new double[HIDDEN_STATE_SIZE];

        for (int t = 1; t < SAMPLE_LENGTH; t++) {
            double[] x = embeddingMatrix[seedIdx];
            double[] newHidden = new double[HIDDEN_STATE_SIZE];
            for (int i = 0; i < HIDDEN_STATE_SIZE; i++) {
                double sum = hiddenBias[i];
                for (int j = 0; j < EMBEDDING_DIMENSION; j++) sum += inputToHiddenWeights[i][j] * x[j];
                for (int j = 0; j < HIDDEN_STATE_SIZE; j++) sum += hiddenToHiddenWeights[i][j] * hidden[j];
                newHidden[i] = Math.tanh(sum);
            }
            hidden = newHidden;

            double[] logits = new double[vocabularySize];
            for (int i = 0; i < vocabularySize; i++) {
                double sum = outputBias[i];
                for (int j = 0; j < HIDDEN_STATE_SIZE; j++) sum += hiddenToOutputWeights[i][j] * hidden[j];
                logits[i] = sum;
            }

            double[] probs = softmax(logits);
            seedIdx = randomChoice(probs);
            sb.append(indexToChar.get(seedIdx));
        }

        return sb.toString();
    }

    private double[] softmax(double[] logits) {
        double max = Arrays.stream(logits).max().orElse(0.0);
        double sum = 0.0;
        double[] out = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            out[i] = Math.exp(logits[i] - max);
            sum += out[i];
        }
        for (int i = 0; i < out.length; i++) out[i] /= sum;
        return out;
    }
    public String respondToPrompt(String prompt, int maxLength) {
        double[] hidden = new double[HIDDEN_STATE_SIZE];

        // 1. Prime the hidden state with the prompt
        for (char c : prompt.toCharArray()) {
            Integer idx = charToIndex.get(c);
            if (idx == null) continue;

            double[] x = embeddingMatrix[idx];
            double[] newHidden = new double[HIDDEN_STATE_SIZE];

            for (int i = 0; i < HIDDEN_STATE_SIZE; i++) {
                double sum = hiddenBias[i];
                for (int j = 0; j < EMBEDDING_DIMENSION; j++)
                    sum += inputToHiddenWeights[i][j] * x[j];
                for (int j = 0; j < HIDDEN_STATE_SIZE; j++)
                    sum += hiddenToHiddenWeights[i][j] * hidden[j];
                newHidden[i] = Math.tanh(sum);
            }
            hidden = newHidden;
        }

        // 2. Generate continuation
        StringBuilder out = new StringBuilder();
        int lastIdx = charToIndex.getOrDefault(
                prompt.charAt(prompt.length() - 1),
                random.nextInt(vocabularySize)
        );

        for (int t = 0; t < maxLength; t++) {
            double[] logits = new double[vocabularySize];
            for (int i = 0; i < vocabularySize; i++) {
                double sum = outputBias[i];
                for (int j = 0; j < HIDDEN_STATE_SIZE; j++)
                    sum += hiddenToOutputWeights[i][j] * hidden[j];
                logits[i] = sum;
            }

            double[] probs = softmax(logits);
            lastIdx = randomChoice(probs);
            char nextChar = indexToChar.get(lastIdx);
            out.append(nextChar);

            // advance hidden state
            double[] x = embeddingMatrix[lastIdx];
            double[] newHidden = new double[HIDDEN_STATE_SIZE];
            for (int i = 0; i < HIDDEN_STATE_SIZE; i++) {
                double sum = hiddenBias[i];
                for (int j = 0; j < EMBEDDING_DIMENSION; j++)
                    sum += inputToHiddenWeights[i][j] * x[j];
                for (int j = 0; j < HIDDEN_STATE_SIZE; j++)
                    sum += hiddenToHiddenWeights[i][j] * hidden[j];
                newHidden[i] = Math.tanh(sum);
            }
            hidden = newHidden;
        }

        return out.toString();
    }


    private int randomChoice(double[] probs) {
        double r = random.nextDouble();
        double cum = 0.0;
        for (int i = 0; i < probs.length; i++) {
            cum += probs[i];
            if (r < cum) return i;
        }
        return probs.length - 1;
    }

    // ===== Save / Load =====
    private void saveModel(String filename) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeObject(this);
            System.out.println("Model saved to " + filename);
        } catch (IOException e) {
            System.out.println("Error saving model: " + e.getMessage());
        }
    }

    private boolean loadModel(String filename) {
        Path path = Paths.get(filename);
        if (!Files.exists(path)) return false;
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            FriendlyLLM loaded = (FriendlyLLM) ois.readObject();
            this.embeddingMatrix = loaded.embeddingMatrix;
            this.inputToHiddenWeights = loaded.inputToHiddenWeights;
            this.hiddenToHiddenWeights = loaded.hiddenToHiddenWeights;
            this.hiddenBias = loaded.hiddenBias;
            this.hiddenToOutputWeights = loaded.hiddenToOutputWeights;
            this.outputBias = loaded.outputBias;
            this.vocabularyList = loaded.vocabularyList;
            this.charToIndex = loaded.charToIndex;
            this.indexToChar = loaded.indexToChar;
            this.vocabularySize = loaded.vocabularySize;
            this.random = new Random(RANDOM_SEED);
            this.threadPool = new ForkJoinPool();
            return true;
        } catch (Exception e) {
            System.out.println("Error loading model: " + e.getMessage());
            return false;
        }
    }

    private void resetModel(String filename) {
        try {
            Files.deleteIfExists(Paths.get(filename));
            initializeModelParameters();
            System.out.println("Model reset. Delete " + filename + " completed.");
        } catch (IOException e) {
            System.out.println("Reset failed: " + e.getMessage());
        }
    }

    private void printInfo() {
        System.out.println("=== MODEL INFO ===");
        System.out.println("Vocabulary size: " + vocabularySize);
        System.out.println("Embedding dimension: " + EMBEDDING_DIMENSION);
        System.out.println("Hidden state size: " + HIDDEN_STATE_SIZE);
        System.out.println("Sequence length: " + SEQUENCE_LENGTH);
        System.out.println("Total characters in training text: " + trainingText.length());
        System.out.println("Random seed: " + RANDOM_SEED);
        System.out.println("==================");
    }


    // ===== Utilities =====
    private static double[][] randomMatrix(int rows, int cols, double scale) {
        Random r = new Random(RANDOM_SEED ^ 0x123456);
        double[][] m = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i][j] = r.nextGaussian() * scale;
        return m;
    }

    // ===== Inner Classes =====
    private static class ForwardCache implements Serializable {
        double[][] hiddenStates;
        double[][] probabilities;

        ForwardCache(int seqLen, int hiddenSize, int vocabSize) {
            hiddenStates = new double[seqLen][hiddenSize];
            probabilities = new double[seqLen][vocabSize];
        }
    }

    private class Gradients implements Serializable {
        double[][] dEmbedding = new double[vocabularySize][EMBEDDING_DIMENSION];
        double[][] dInputToHidden = new double[HIDDEN_STATE_SIZE][EMBEDDING_DIMENSION];
        double[][] dHiddenToHidden = new double[HIDDEN_STATE_SIZE][HIDDEN_STATE_SIZE];
        double[] dHiddenBias = new double[HIDDEN_STATE_SIZE];
        double[][] dHiddenToOutput = new double[vocabularySize][HIDDEN_STATE_SIZE];
        double[] dOutputBias = new double[vocabularySize];
    }
}
