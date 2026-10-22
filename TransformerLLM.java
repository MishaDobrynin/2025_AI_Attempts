import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;

public class TransformerLLM implements Serializable {
    private static final long serialVersionUID = 1L;

    // ===== Hyperparameters =====
    private static final int EMBEDDING_DIM = 64;
    private static final int SEQ_LEN = 50;
    private static final double LR = 0.005;
    private static final double GRAD_CLIP = 5.0;
    private static final int SAMPLE_LEN = 200;
    private static final long RANDOM_SEED = 42L;

    // ===== Model parameters =====
    private double[][] embeddings;
    private double[][] Wq, Wk, Wv, Wo;
    private double[] outputBias;

    // ===== Data =====
    private String trainingText;
    private List<Character> vocabList;
    private Map<Character, Integer> charToIndex;
    private Map<Integer, Character> indexToChar;
    private int vocabSize;

    private transient Random random;
    private transient ForkJoinPool threadPool;

    public TransformerLLM() {
        random = new Random(RANDOM_SEED);
        threadPool = new ForkJoinPool();
    }

    public static void main(String[] args) throws Exception {
        TransformerLLM model = new TransformerLLM();
        model.loadOrCreateData("data.txt");

        boolean loaded = model.loadModel("transformer.dat");
        if (!loaded) {
            model.initializeParameters();
        }

        Scanner scanner = new Scanner(System.in);
        boolean run = true;
        while (run) {
            System.out.println("Action? (y=train, p=prompt, s=save, l=load, r=reset, i=info, q=quit)");
            String input = scanner.nextLine().trim();
            switch (input) {
                case "y" -> model.continuousTrain();
                case "p" -> {
                    System.out.print("Prompt: ");
                    String prompt = scanner.nextLine();
                    System.out.println("AI:");
                    System.out.println(model.generateFromPrompt(prompt, SAMPLE_LEN));
                }
                case "s" -> model.saveModel("transformer.dat");
                case "l" -> model.loadModel("transformer.dat");
                case "r" -> model.resetModel("transformer.dat");
                case "i" -> model.printInfo();
                case "q" -> {
                    model.saveModel("transformer.dat");
                    run = false;
                }
                default -> System.out.println("Unknown command.");
            }
        }
    }

    // ===== Data Loading =====
    private void loadOrCreateData(String filename) throws IOException {
        Path path = Paths.get(filename);
        if (!Files.exists(path)) throw new FileNotFoundException(filename + " not found.");
        trainingText = new String(Files.readAllBytes(path)).replace("\r", "");
        buildVocab();
    }

    private void buildVocab() {
        Set<Character> vocabSet = new LinkedHashSet<>();
        for (char c : trainingText.toCharArray()) vocabSet.add(c);
        vocabList = new ArrayList<>(vocabSet);
        vocabSize = vocabList.size();
        charToIndex = new HashMap<>();
        indexToChar = new HashMap<>();
        for (int i = 0; i < vocabSize; i++) {
            charToIndex.put(vocabList.get(i), i);
            indexToChar.put(i, vocabList.get(i));
        }
    }

    // ===== Model Initialization =====
    private void initializeParameters() {
        embeddings = randomMatrix(vocabSize, EMBEDDING_DIM, 0.01);
        Wq = randomMatrix(EMBEDDING_DIM, EMBEDDING_DIM, 0.1);
        Wk = randomMatrix(EMBEDDING_DIM, EMBEDDING_DIM, 0.1);
        Wv = randomMatrix(EMBEDDING_DIM, EMBEDDING_DIM, 0.1);
        Wo = randomMatrix(EMBEDDING_DIM, EMBEDDING_DIM, 0.1);
        outputBias = new double[vocabSize];
    }

    // ===== Continuous Training =====
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
        int pointer = 0;
        List<Callable<Void>> tasks = new ArrayList<>();

        while (pointer + SEQ_LEN + 1 <= trainingText.length()) {
            int start = pointer;
            tasks.add(() -> {
                trainSequence(start);
                return null;
            });
            pointer += SEQ_LEN;
        }
        threadPool.invokeAll(tasks);
    }

    private void trainSequence(int start) {
        int[] inputIndices = new int[SEQ_LEN];
        int[] targetIndices = new int[SEQ_LEN];

        for (int t = 0; t < SEQ_LEN; t++) {
            inputIndices[t] = charToIndex.get(trainingText.charAt(start + t));
            targetIndices[t] = charToIndex.get(trainingText.charAt(start + t + 1));
        }

        double[][] X = new double[SEQ_LEN][EMBEDDING_DIM];
        for (int t = 0; t < SEQ_LEN; t++) X[t] = embeddings[inputIndices[t]];

        double[][] attentionOutput = forward(X);
        backward(attentionOutput, inputIndices, targetIndices);
    }

    // ===== Forward / Attention =====
    private double[][] forward(double[][] X) {
        int L = X.length;
        double[][] Q = matMul(X, Wq);
        double[][] K = matMul(X, Wk);
        double[][] V = matMul(X, Wv);

        double[][] scores = new double[L][L];
        for (int i = 0; i < L; i++)
            for (int j = 0; j < L; j++)
                scores[i][j] = dot(Q[i], K[j]) / Math.sqrt(EMBEDDING_DIM);

        double[][] attention = softmax2D(scores);
        double[][] output = new double[L][EMBEDDING_DIM];
        for (int i = 0; i < L; i++)
            for (int j = 0; j < L; j++)
                for (int k = 0; k < EMBEDDING_DIM; k++)
                    output[i][k] += attention[i][j] * V[j][k];

        output = matMul(output, Wo);
        return output;
    }

    private void backward(double[][] output, int[] inputIndices, int[] targetIndices) {
        double[][] dWo = new double[EMBEDDING_DIM][EMBEDDING_DIM];
        double[][] dEmbeddings = new double[vocabSize][EMBEDDING_DIM];
        double[] dOutputBias = new double[vocabSize];

        // Simple gradient wrt output projection
        for (int t = 0; t < SEQ_LEN; t++) {
            double[] logits = new double[vocabSize];
            for (int i = 0; i < vocabSize; i++)
                for (int j = 0; j < EMBEDDING_DIM; j++)
                    logits[i] += output[t][j] * (i == targetIndices[t] ? 1.0 : 0.0);
            // Backprop simplified for demonstration
            for (int i = 0; i < EMBEDDING_DIM; i++)
                for (int j = 0; j < EMBEDDING_DIM; j++)
                    dWo[i][j] += output[t][i] * output[t][j];
            for (int i = 0; i < EMBEDDING_DIM; i++)
                dEmbeddings[inputIndices[t]][i] += 1.0; // simplified
        }

        // Gradient clipping and update
        for (int i = 0; i < EMBEDDING_DIM; i++)
            for (int j = 0; j < EMBEDDING_DIM; j++)
                Wo[i][j] -= LR * Math.max(Math.min(dWo[i][j], GRAD_CLIP), -GRAD_CLIP);

        for (int i = 0; i < vocabSize; i++)
            for (int j = 0; j < EMBEDDING_DIM; j++)
                embeddings[i][j] -= LR * Math.max(Math.min(dEmbeddings[i][j], GRAD_CLIP), -GRAD_CLIP);
    }
    private static double[] softmax(double[] x) {
        double max = Arrays.stream(x).max().orElse(0);
        double sum = 0;
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) sum += Math.exp(x[i] - max);
        for (int i = 0; i < x.length; i++) y[i] = Math.exp(x[i] - max) / sum;
        return y;
    }


    // ===== Sampling / Prompt Response =====
    public String generateFromPrompt(String prompt, int maxLength) {
        List<Integer> indices = new ArrayList<>();
        for (char c : prompt.toCharArray())
            indices.add(charToIndex.getOrDefault(c, random.nextInt(vocabSize)));

        double[][] X = new double[Math.min(SEQ_LEN, indices.size())][EMBEDDING_DIM];
        for (int t = 0; t < X.length; t++) X[t] = embeddings[indices.get(t)];

        StringBuilder out = new StringBuilder(prompt);

        for (int t = 0; t < maxLength; t++) {
            double[][] attentionOutput = forward(X);
            double[] last = attentionOutput[attentionOutput.length - 1];
            double[] probs = softmax(last);

            int idx = randomChoice(probs);
            out.append(indexToChar.get(idx));

            // Slide window
            double[][] newX = new double[Math.min(SEQ_LEN, X.length + 1)][EMBEDDING_DIM];
            int offset = Math.max(0, X.length + 1 - SEQ_LEN);
            for (int i = 0; i < newX.length - 1; i++) newX[i] = X[i + offset];
            newX[newX.length - 1] = embeddings[idx];
            X = newX;
        }
        return out.toString();
    }

    // ===== Save / Load =====
    private void saveModel(String filename) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeObject(this);
            System.out.println("Model saved to " + filename);
        } catch (IOException e) {
            System.out.println("Save failed: " + e.getMessage());
        }
    }

    private boolean loadModel(String filename) {
        Path path = Paths.get(filename);
        if (!Files.exists(path)) return false;
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            TransformerLLM loaded = (TransformerLLM) ois.readObject();
            this.embeddings = loaded.embeddings;
            this.Wq = loaded.Wq;
            this.Wk = loaded.Wk;
            this.Wv = loaded.Wv;
            this.Wo = loaded.Wo;
            this.outputBias = loaded.outputBias;
            this.vocabList = loaded.vocabList;
            this.charToIndex = loaded.charToIndex;
            this.indexToChar = loaded.indexToChar;
            this.vocabSize = loaded.vocabSize;
            this.random = new Random(RANDOM_SEED);
            this.threadPool = new ForkJoinPool();
            System.out.println("Model loaded from " + filename);
            return true;
        } catch (Exception e) {
            System.out.println("Load failed: " + e.getMessage());
            return false;
        }
    }

    private void resetModel(String filename) {
        try {
            Files.deleteIfExists(Paths.get(filename));
            initializeParameters();
            System.out.println("Model reset.");
        } catch (IOException e) {
            System.out.println("Reset failed: " + e.getMessage());
        }
    }

    // ===== Utilities =====
    private void printInfo() {
        System.out.println("Vocabulary size: " + vocabSize);
        System.out.println("Embedding dimension: " + EMBEDDING_DIM);
        System.out.println("Sequence length: " + SEQ_LEN);
    }

    private static double[][] randomMatrix(int rows, int cols, double scale) {
        Random r = new Random(RANDOM_SEED ^ 0x123456);
        double[][] m = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i][j] = r.nextGaussian() * scale;
        return m;
    }

    private static double[][] matMul(double[][] A, double[][] B) {
        int n = A.length, k = A[0].length, m = B[0].length;
        double[][] C = new double[n][m];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                for (int p = 0; p < k; p++)
                    C[i][j] += A[i][p] * B[p][j];
        return C;
    }

    private static double dot(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }

    private static double[][] softmax2D(double[][] x) {
        int n = x.length, m = x[0].length;
        double[][] y = new double[n][m];
        for (int i = 0; i < n; i++) {
            double max = Arrays.stream(x[i]).max().orElse(0);
            double sum = 0;
            for (int j = 0; j < m; j++) sum += Math.exp(x[i][j] - max);
            for (int j = 0; j < m; j++) y[i][j] = Math.exp(x[i][j] - max) / sum;
        }
        return y;
    }

    private int randomChoice(double[] probs) {
        double r = random.nextDouble();
        double cum = 0;
        for (int i = 0; i < probs.length; i++) {
            cum += probs[i];
            if (r < cum) return i;
        }
        return probs.length - 1;
    }
}
