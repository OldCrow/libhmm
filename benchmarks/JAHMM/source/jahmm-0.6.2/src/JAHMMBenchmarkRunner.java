
import java.io.*;
import java.util.*;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;

import be.ac.ulg.montefiore.run.jahmm.*;
import be.ac.ulg.montefiore.run.jahmm.learn.*;

public class JAHMMBenchmarkRunner {
    
    // Enum to represent discrete observations
    public enum ObsSymbol {
        SYM_0, SYM_1, SYM_2, SYM_3, SYM_4, SYM_5, SYM_6, SYM_7;
        
        public static ObsSymbol fromInteger(int value) {
            switch(value) {
                case 0: return SYM_0;
                case 1: return SYM_1;
                case 2: return SYM_2;
                case 3: return SYM_3;
                case 4: return SYM_4;
                case 5: return SYM_5;
                case 6: return SYM_6;
                case 7: return SYM_7;
                default: throw new IllegalArgumentException("Invalid observation value: " + value);
            }
        }
    }
    
    private static class HMMConfig {
        int numStates;
        int alphabetSize;
        double[] initialProbs;
        double[][] transitionMatrix;
        double[][] emissionMatrix;
        
        public static HMMConfig loadFromFile(String configFile) throws IOException {
            Properties props = new Properties();
            props.load(new FileInputStream(configFile));
            
            HMMConfig config = new HMMConfig();
            config.numStates = Integer.parseInt(props.getProperty("num_states"));
            config.alphabetSize = Integer.parseInt(props.getProperty("alphabet_size"));
            
            // Parse initial probabilities
            String[] initialProbsStr = props.getProperty("initial_probs").split(",");
            config.initialProbs = new double[config.numStates];
            for (int i = 0; i < config.numStates; i++) {
                config.initialProbs[i] = Double.parseDouble(initialProbsStr[i]);
            }
            
            // Parse transition matrix
            config.transitionMatrix = new double[config.numStates][config.numStates];
            for (int i = 0; i < config.numStates; i++) {
                String[] transitionStr = props.getProperty("transition_" + i).split(",");
                for (int j = 0; j < config.numStates; j++) {
                    config.transitionMatrix[i][j] = Double.parseDouble(transitionStr[j]);
                }
            }
            
            // Parse emission matrix
            config.emissionMatrix = new double[config.numStates][config.alphabetSize];
            for (int i = 0; i < config.numStates; i++) {
                String[] emissionStr = props.getProperty("emission_" + i).split(",");
                for (int j = 0; j < config.alphabetSize; j++) {
                    config.emissionMatrix[i][j] = Double.parseDouble(emissionStr[j]);
                }
            }
            
            return config;
        }
    }
    
    private static class TimingResult {
        double forwardTime;
        double viterbiTime;
        double likelihood;
    }
    
    public static void main(String[] args) {
        if (args.length != 3) {
            System.err.println("Usage: JAHMMBenchmarkRunner <config_file> <obs_file> <result_file>");
            System.exit(1);
        }
        
        try {
            String configFile = args[0];
            String obsFile = args[1];
            String resultFile = args[2];
            
            // Load configuration
            HMMConfig config = HMMConfig.loadFromFile(configFile);
            
            // Load observations
            List<Integer> observations = new ArrayList<>();
            Scanner scanner = new Scanner(new File(obsFile));
            while (scanner.hasNextInt()) {
                observations.add(scanner.nextInt());
            }
            scanner.close();
            
            // Build HMM
            Hmm<ObservationDiscrete<ObsSymbol>> hmm = buildHMM(config);
            
            // Convert observations to JAHMM format
            List<ObservationDiscrete<ObsSymbol>> observationSequence = new ArrayList<>();
            for (Integer obs : observations) {
                observationSequence.add(new ObservationDiscrete<ObsSymbol>(ObsSymbol.fromInteger(obs)));
            }
            
            // Run benchmarks
            TimingResult result = runBenchmarks(hmm, observationSequence);
            
            // Write results
            PrintWriter writer = new PrintWriter(new FileWriter(resultFile));
            writer.println("FORWARD_TIME:" + result.forwardTime);
            writer.println("VITERBI_TIME:" + result.viterbiTime);
            writer.println("LIKELIHOOD:" + result.likelihood);
            writer.close();
            
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    private static Hmm<ObservationDiscrete<ObsSymbol>> buildHMM(HMMConfig config) {
        // Create HMM with discrete observations
        Hmm<ObservationDiscrete<ObsSymbol>> hmm = 
            new Hmm<ObservationDiscrete<ObsSymbol>>(config.numStates, 
                new OpdfDiscreteFactory<ObsSymbol>(ObsSymbol.class));
        
        // Set initial probabilities
        for (int i = 0; i < config.numStates; i++) {
            hmm.setPi(i, config.initialProbs[i]);
        }
        
        // Set transition probabilities
        for (int i = 0; i < config.numStates; i++) {
            for (int j = 0; j < config.numStates; j++) {
                hmm.setAij(i, j, config.transitionMatrix[i][j]);
            }
        }
        
        // Set emission probabilities
        for (int i = 0; i < config.numStates; i++) {
            OpdfDiscrete<ObsSymbol> opdf = new OpdfDiscrete<ObsSymbol>(ObsSymbol.class, config.emissionMatrix[i]);
            hmm.setOpdf(i, opdf);
        }
        
        return hmm;
    }
    
    private static TimingResult runBenchmarks(Hmm<ObservationDiscrete<ObsSymbol>> hmm, 
                                            List<ObservationDiscrete<ObsSymbol>> observations) {
        TimingResult result = new TimingResult();
        
        // Benchmark Forward-Backward algorithm
        long startTime = System.nanoTime();
        result.likelihood = hmm.probability(observations);
        long endTime = System.nanoTime();
        result.forwardTime = (endTime - startTime) / 1000000.0; // Convert to milliseconds
        
        // Benchmark Viterbi algorithm
        startTime = System.nanoTime();
        int[] viterbiPath = hmm.mostLikelyStateSequence(observations);
        endTime = System.nanoTime();
        result.viterbiTime = (endTime - startTime) / 1000000.0; // Convert to milliseconds
        
        return result;
    }
}
