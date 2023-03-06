/* Copyright (c) 2010-2012 Sebastian Bauer
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the
 * disclaimer below) provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the
 *   distribution.
 *
 * * Neither the name of Sebastian Bauer nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
 * GRANTED BY THIS LICENSE.  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
 * HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package sonumina.boqa.benchmark;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import ontologizer.dotwriter.AbstractDotAttributesProvider;
import ontologizer.dotwriter.GODOTWriter;
import ontologizer.go.Ontology;
import ontologizer.go.Term;
import ontologizer.go.TermID;
import sonumina.boqa.calculation.BOQA;
import sonumina.boqa.calculation.BOQA.Result;
import sonumina.boqa.calculation.Observations;
import sonumina.math.graph.SlimDirectedGraphView;

/**
 * Class that implements the logic of the benchmark presented in the paper.
 *
 * @author Sebastian Bauer
 */
public class Benchmark
{
    private Ontology graph;

    private SlimDirectedGraphView<Term> slimGraph;

    private BOQA boqa;

    /** Verbose output */
    private boolean VERBOSE;

    /** Use threading */
    private final boolean THREADING_IN_SIMULATION = true;

    /** The full name (including txt suffix) of the results to be written */
    private String RESULT_NAME = "benchmark.txt";

    /** Number of samples taken per item */
    private int samplesPerItem = 5;

    /**
     * Container for a full experiment. Contains input data as well as results.
     *
     * @author Sebastian Bauer
     */
    static class ExperimentStore
    {
        Observations obs;

        Result modelWithoutFrequencies;

        Result modelWithFrequencies;

        Result resnik;

        Result lin;

        Result jc;

        Result mb;
    }

    /**
     * Sets the base name of the results to be written.
     */
    public void setResultBaseName(String name)
    {
        this.RESULT_NAME = name + ".txt";
    }

    /**
     * Sets the samples that are generated for each item during the simulation.
     *
     * @param samplesPerItem
     */
    public void setSamplesPerItem(int samplesPerItem)
    {
        this.samplesPerItem = samplesPerItem;
    }

    /**
     * Processes the simulation and evaluation for the given item. That is, we first generate some obfuscated
     * observations and then apply the tested algorithm to recover the signal.
     *
     * @param item
     * @returns an ExperimentStore
     */
    private ExperimentStore processItem(int item, boolean provideGraph, Random rnd)
    {
        int i;

        Observations obs = this.boqa.generateObservations(item, rnd);

        boolean[] observations = obs.observations;

        /* First, without taking frequencies into account */
        Result modelWithoutFrequencies = this.boqa.assignMarginals(obs, false);

        /* Second, with taking frequencies into account */
        Result modelWithFrequencies = this.boqa.assignMarginals(obs, true);

        ExperimentStore id = new ExperimentStore();
        id.obs = obs;
        id.modelWithoutFrequencies = modelWithoutFrequencies;
        id.modelWithFrequencies = modelWithFrequencies;
        id.resnik = this.boqa.resnikScore(obs.observations, true, rnd);
        id.lin = this.boqa.linScore(obs.observations, true, rnd);
        id.jc = this.boqa.jcScore(obs.observations, true, rnd);
        id.mb = this.boqa.mbScore(obs.observations);

        /******** The rest is for debugging purposes ********/
        if (this.VERBOSE || provideGraph) {
            class Pair implements Comparable<Pair>
            {
                double score;

                int idx;

                public Pair(int idx, double score)
                {
                    this.idx = idx;
                    this.score = score;
                }

                @Override
                public int compareTo(Pair o)
                {
                    if (this.score < o.score) {
                        return 1;
                    } else if (this.score > o.score) {
                        return -1;
                    }
                    return 0;
                };

            }
            ;

            ArrayList<Pair> scoreList = new ArrayList<Pair>(this.boqa.allItemList.size());
            ArrayList<Pair> idealList = new ArrayList<Pair>(this.boqa.allItemList.size());
            for (i = 0; i < this.boqa.allItemList.size(); i++) {
                scoreList.add(new Pair(i, modelWithoutFrequencies.getScore(i)));
                idealList.add(new Pair(i, modelWithoutFrequencies.getMarginalIdeal(i)));
            }
            Collections.sort(scoreList);
            Collections.sort(idealList, new Comparator<Pair>()
            {
                @Override
                public int compare(Pair o1, Pair o2)
                {
                    if (o1.score > o2.score) {
                        return 1;
                    }
                    if (o1.score < o2.score) {
                        return -1;
                    }
                    return 0;
                };
            });

            /* Display top 10 */
            for (i = 0; i < Math.min(10, scoreList.size()); i++) {
                Pair p = scoreList.get(i);
                boolean itIs = p.idx == item;
                System.out.println((i + 1) + (itIs ? "(*)" : "") + ": " + this.boqa.allItemList.get(p.idx) + ": "
                    + p.score
                    + " " + modelWithoutFrequencies.getMarginal(p.idx));
            }

            int scoreRank = 0;
            int marginalIdealRank = 0;

            /* And where the searched item is */
            for (i = 0; i < scoreList.size(); i++) {
                Pair p = scoreList.get(i);
                // boolean itIs = p.idx == item;
                if (p.idx == item) {
                    scoreRank = i + 1;
                    break;
                }
            }

            for (i = 0; i < idealList.size(); i++) {
                Pair p = scoreList.get(i);
                if (p.idx == item) {
                    marginalIdealRank = i + 1;
                    break;
                }
            }

            // System.out.println((i+1) + (itIs?"(*)":"") + ": " + allItemList.get(p.idx) + ": " + p.score + " " +
            // modelWithoutFrequencies.getMarginal(p.idx));
            System.out.println("Rank of searched item. Score: " + scoreRank + "  Ideal: " + marginalIdealRank + " ( "
                + modelWithoutFrequencies.getMarginalIdeal(item) + ")");

            System.out.println("Statistics of the searched item");
            System.out.println(modelWithoutFrequencies.getStats(item).toString());
            System.out.println("Statistics for the top item");
            System.out.println(modelWithoutFrequencies.getStats(scoreList.get(0).idx).toString());

            // for (i=0;i<Stats.NodeCase.values().length;i++)
            // System.out.println(" " + Stats.NodeCase.values()[i].name() + ": " +
            // modelWithoutFrequencies.statsMatrix[item][i]);
            //
            // System.out.println("Statistics for the top item");
            // for (i=0;i<Stats.NodeCase.values().length;i++)
            // System.out.println(" " + Stats.NodeCase.values()[i].name() + ": " +
            // modelWithoutFrequencies.stateMatrix[scoreList.get(0).idx][i]);

            if (provideGraph) {
                /* Output the graph */
                final HashSet<TermID> hiddenSet = new HashSet<TermID>();
                for (i = 0; i < this.boqa.getTermsDirectlyAnnotatedTo(item).length; i++) {
                    hiddenSet.add(this.slimGraph.getVertex(this.boqa.getTermsDirectlyAnnotatedTo(item)[i]).getID());
                }
                final HashSet<TermID> observedSet = new HashSet<TermID>();
                for (i = 0; i < observations.length; i++) {
                    if (observations[i]) {
                        observedSet.add(this.slimGraph.getVertex(i).getID());
                    }
                }
                int topRankIdx = scoreList.get(0).idx;
                final HashSet<TermID> topRankSet = new HashSet<TermID>();
                for (i = 0; i < this.boqa.getTermsDirectlyAnnotatedTo(topRankIdx).length; i++) {
                    topRankSet.add(this.slimGraph.getVertex(this.boqa.getTermsDirectlyAnnotatedTo(topRankIdx)[i])
                        .getID());
                }

                HashSet<TermID> allSet = new HashSet<TermID>();
                allSet.addAll(hiddenSet);
                allSet.addAll(observedSet);
                allSet.addAll(topRankSet);
                GODOTWriter.writeDOT(this.graph, new File("setting.dot"), null, allSet,
                    new AbstractDotAttributesProvider()
                    {
                        @Override
                        public String getDotNodeAttributes(TermID id)
                        {
                            String fillcolor = "";
                            String label = Benchmark.this.graph.getTerm(id).getName();
                            String flags = "";
                            if (hiddenSet.contains(id)) {
                                fillcolor = ",style=filled,fillcolor=gray";
                            }

                            if (topRankSet.contains(id)) {
                                flags += "1";
                            }
                            if (observedSet.contains(id)) {
                                flags += "O";
                            }

                            if (flags.length() != 0) {
                                label += "(" + flags + ")";
                            }

                            return "label=\"" + label + "\"" + fillcolor;
                        }
                    });
            }
        }

        return id;
    }

    /**
     * Perform the benchmark as described in the paper on the given BOQA context. Produces a bunch of files in the
     * current directory.
     *
     * @param args
     * @throws InterruptedException
     * @throws IOException
     */
    @SuppressWarnings("unused")
    public void benchmark(BOQA boqa) throws InterruptedException, IOException
    {
        int i;
        int numProcessors = BOQA.getNumProcessors();

        /* TODO: Get rid of this ugliness */
        this.boqa = boqa;

        double ALPHA = boqa.getSimulationAlpha();
        double BETA = boqa.getSimulationBeta();
        int maxTerms = boqa.getSimulationMaxTerms();
        boolean CONSIDER_FREQUENCIES_ONLY = boqa.getConsiderFrequenciesOnly();

        this.graph = boqa.getOntology();
        this.slimGraph = boqa.getSlimGraph();

        /**************************************************************************************************************************/
        /* Write score distribution */

        if (false) {
            boqa.writeScoreDistribution(new File("score-0.txt"), 0);
        }

        /**************************************************************************************************************************/

        /* Write example */

        HashSet<TermID> hpoTerms = new HashSet<TermID>();
        hpoTerms.add(new TermID("HP:0000822")); /* Hypertension */
        hpoTerms.add(new TermID("HP:0000875")); /* Episodic Hypertension */
        hpoTerms.add(new TermID("HP:0002621")); /* Atherosclerosis */

        boqa.writeDOTExample(new File("hpo-example.dot"), hpoTerms);

        /**************************************************************************************************************************/

        int firstItemWithFrequencies = -1;
        int numItemsWithFrequencies = 0;
        for (i = 0; i < boqa.getNumberOfItems(); i++) {
            if (boqa.hasItemFrequencies(i)) {
                numItemsWithFrequencies++;
                if (firstItemWithFrequencies == -1) {
                    firstItemWithFrequencies = i;
                }
            }
        }

        System.out.println("Items with frequencies " + numItemsWithFrequencies + "  First one: " +
            firstItemWithFrequencies + " which is "
            + (firstItemWithFrequencies != -1 ? boqa.allItemList.get(firstItemWithFrequencies) : ""));

        /**************************************************************************************************************************/

        String evidenceString = "All";
        String[] evidenceCodes = boqa.getEvidenceCodes();
        if (evidenceCodes != null && evidenceCodes.length > 0) {
            StringBuilder evidenceBuilder = new StringBuilder();
            evidenceBuilder.append("\"");
            evidenceBuilder.append(evidenceCodes[0]);

            for (String evidenceCode : evidenceCodes) {
                evidenceBuilder.append("," + evidenceCode);
            }
            evidenceString = evidenceBuilder.toString();
        }

        /* Remember the parameter */
        BufferedWriter param = new BufferedWriter(new FileWriter(this.RESULT_NAME.split("\\.")[0] + "_param.txt"));
        param
            .write(
                "alpha\tbeta\tconsider.freqs.only\titems\tterms\tmax.terms\tmax.samples\tevidences\tmax.freq.terms\n");
        param.write(String.format("%g\t%g\t%b\t%d\t%d\t%d\t%d\t%s\t%d\n", ALPHA, BETA, CONSIDER_FREQUENCIES_ONLY,
            boqa.getNumberOfItems(), this.slimGraph.getNumberOfVertices(), maxTerms, this.samplesPerItem,
            evidenceString,
            boqa.getMaxFrequencyTerms()));
        param.flush();
        param.close();

        /* Write out r code to load matrix in */
        BufferedWriter load = new BufferedWriter(new FileWriter(this.RESULT_NAME.split("\\.")[0] + "_load.R"));
        load.append("boqa.load.data<-function() {\n d<-read.table(");
        load.append("\"" + new File(this.RESULT_NAME).getAbsolutePath() + "\", ");
        load.append("colClasses=c(\"integer\",\"integer\",rep(\"numeric\",13),\"integer\"),h=F");
        load.append(")");
        load.append(
            "\n colnames(d)<-c(\"run\",\"label\",\"score\",\"marg\",\"marg.ideal\", \"score.freq\",\"marg.freq\", \"marg.freq.ideal\", \"resnik.avg\", \"resnik.avg.p\", \"lin.avg\", \"lin.avg.p\", \"jc.avg\", \"jc.avg.p\", \"mb\", \"freq\");");
        load.append("\n return (d);");
        load.append("\n}\n");
        load.append("boqa.name<-\"");
        load.append(new File(this.RESULT_NAME).getAbsolutePath());
        load.append("\";\n");
        load.append("boqa.base.name<-\"");
        load.append(new File(this.RESULT_NAME.split("\\.")[0]).getAbsolutePath());
        load.append("\";\n");
        load.flush();
        load.close();

        final BufferedWriter out = new BufferedWriter(new FileWriter(this.RESULT_NAME));
        final BufferedWriter summary =
            new BufferedWriter(new FileWriter(this.RESULT_NAME.split("\\.")[0] + "_summary.txt"));

        ExecutorService es = Executors.newFixedThreadPool(numProcessors);

        Random rnd = new Random(9);

        int run = 0;

        for (int sample = 0; sample < this.samplesPerItem; sample++) {
            for (i = 0; i < boqa.getNumberOfItems(); i++) {
                final long seed = rnd.nextLong();
                final int item = i;
                final int fixedRun = run++;

                Runnable thread = new Runnable()
                {
                    @Override
                    public void run()
                    {
                        StringBuilder resultBuilder = new StringBuilder();

                        System.out.println("Seed = " + seed + " run = " + fixedRun);

                        ExperimentStore store = processItem(item, false, new Random(seed));

                        for (int j = 0; j < Benchmark.this.boqa.getNumberOfItems(); j++) {
                            resultBuilder.append(fixedRun);
                            resultBuilder.append("\t");
                            resultBuilder.append(item == j ? 1 : 0);
                            resultBuilder.append("\t");
                            resultBuilder.append(store.modelWithoutFrequencies.getScore(j));
                            resultBuilder.append("\t");
                            resultBuilder.append(store.modelWithoutFrequencies.getMarginal(j));
                            resultBuilder.append("\t");
                            resultBuilder.append(store.modelWithoutFrequencies.getMarginalIdeal(j));
                            resultBuilder.append("\t");
                            resultBuilder.append(store.modelWithFrequencies.getScore(j));
                            resultBuilder.append("\t");
                            resultBuilder.append(store.modelWithFrequencies.getMarginal(j));
                            resultBuilder.append("\t");
                            resultBuilder.append(store.modelWithFrequencies.getMarginalIdeal(j));
                            resultBuilder.append("\t");
                            resultBuilder.append(store.resnik.getScore(j));
                            resultBuilder.append("\t");
                            resultBuilder.append(store.resnik.getMarginal(j));
                            resultBuilder.append("\t");
                            resultBuilder.append(store.lin.getScore(j));
                            resultBuilder.append("\t");
                            resultBuilder.append(store.lin.getMarginal(j));
                            resultBuilder.append("\t");
                            resultBuilder.append(store.jc.getScore(j));
                            resultBuilder.append("\t");
                            resultBuilder.append(store.jc.getMarginal(j));
                            resultBuilder.append("\t");
                            resultBuilder.append(store.mb.getScore(j));
                            resultBuilder.append("\t");
                            resultBuilder.append(Benchmark.this.boqa.hasItemFrequencies(item) ? 1 : 0);
                            resultBuilder.append("\n");
                        }

                        synchronized (out) {
                            try {
                                out.append(resultBuilder.toString());
                            } catch (IOException e) {
                                // TODO Auto-generated catch block
                                e.printStackTrace();
                            }
                        }

                        String sum =
                            fixedRun + "\t" + store.obs.observationStats.falsePositiveRate() + "\t"
                                + store.obs.observationStats.falseNegativeRate() + "\n";

                        synchronized (summary) {
                            try {
                                summary.write(sum);
                            } catch (IOException e) {
                                // TODO Auto-generated catch block
                                e.printStackTrace();
                            }
                        }
                    }
                };

                if (this.THREADING_IN_SIMULATION) {
                    es.execute(thread);
                } else {
                    thread.run();
                }
            }
        }

        es.shutdown();
        while (!es.awaitTermination(10, TimeUnit.SECONDS)) {
            ;
        }

        synchronized (out) {
            out.close();
        }

        synchronized (summary) {
            summary.close();
        }
    }
}
