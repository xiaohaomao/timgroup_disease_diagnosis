package sonumina.boqa.tests;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.HashSet;
import java.util.Random;

import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ontologizer.association.Association;
import ontologizer.association.AssociationContainer;
import ontologizer.benchmark.Datafiles;
import ontologizer.go.OBOParser;
import ontologizer.go.OBOParserException;
import ontologizer.go.Ontology;
import ontologizer.go.Term;
import ontologizer.go.TermContainer;
import ontologizer.types.ByteString;
import sonumina.boqa.benchmark.Benchmark;
import sonumina.boqa.calculation.BOQA;
import sonumina.boqa.calculation.BOQA.Result;
import sonumina.boqa.calculation.Observations;
import sonumina.math.graph.AbstractGraph.DotAttributesProvider;
import sonumina.math.graph.SlimDirectedGraphView;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class BOQATest
{
    private static Datafiles hpo;

    private Logger logger = LoggerFactory.getLogger(BOQATest.class);

    @BeforeClass
    public static void loadHPO() throws InterruptedException, IOException, URISyntaxException
    {
        hpo = new Datafiles(
            new File(ClassLoader.getSystemResource("human-phenotype-ontology.obo.gz").toURI()).getCanonicalPath(),
            new File(ClassLoader.getSystemResource("phenotype_annotation.omim.gz").toURI()).getCanonicalPath());
    }

    /**
     * Tests the choose function.
     */
    @Test
    public void testChoose()
    {
        Random rnd = new Random();

        /** Create storage where to choose numbers from */
        int[] storage = new int[1000];
        for (int i = 0; i < storage.length; i++) {
            storage[i] = i;
        }

        int[] chosen = new int[10];

        for (int s = 1; s < 10; s++) {
            for (int t = 0; t < 10; t++) {
                BOQA.choose(rnd, s, chosen, storage);

                /* Check storage array for validity */
                boolean[] seen = new boolean[storage.length];
                for (int i = 0; i < storage.length; i++) {
                    assertEquals(false, seen[storage[i]]);
                    seen[storage[i]] = true;
                }

                /* Check chosen array for validity */
                for (int i = 0; i < seen.length; i++) {
                    seen[i] = false;
                }

                for (int i = 0; i < s; i++) {
                    assertEquals(false, seen[chosen[i]]);
                    seen[chosen[i]] = true;
                }
            }
        }
    }

    /**
     * Checks whether all elements of expected are contained in actual and vice versa.
     *
     * @param expected
     * @param actual
     */
    private void checkIntArrayContentsUnordered(int[] expected, int[] actual)
    {
        HashSet<Integer> expectedSet = new HashSet<Integer>();
        HashSet<Integer> actualSet = new HashSet<Integer>();

        assertEquals(expected.length, actual.length);

        for (int e : expected) {
            expectedSet.add(e);
        }

        for (int a : actual) {
            actualSet.add(a);
        }

        assertTrue(expectedSet.containsAll(actualSet));
        assertTrue(actualSet.containsAll(expectedSet));
    }

    @Test
    public void testMostSpecificTerms()
    {
        InternalDatafiles data = new InternalDatafiles();
        BOQA boqa = new BOQA();
        boqa.setConsiderFrequenciesOnly(false);
        boqa.setCacheScoreDistribution(false);
        boqa.setPrecalculateItemMaxs(false);
        boqa.setPrecalculateScoreDistribution(false);
        boqa.setup(data.graph, data.assoc);

        checkIntArrayContentsUnordered(new int[] { 1, 2 }, boqa.mostSpecificTerms(new int[] { 0, 1, 2 }));
        checkIntArrayContentsUnordered(new int[] { 9, 10, 11, 12, 13, 14 },
            boqa.mostSpecificTerms(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 }));
    }

    /**
     * A helper function to test similarity values of the internal ontology.
     *
     * @param boqa
     */
    private void checkInternalSimValues(BOQA boqa)
    {
        /* Common ancestors */
        assertEquals(7, boqa.getCommonAncestorWithMaxIC(11, 12));
        assertTrue(boqa.getCommonAncestorWithMaxIC(9, 11) == 4 || boqa.getCommonAncestorWithMaxIC(9, 11) == 6);

        /* Resnik */
        assertEquals(0.223144, boqa.resScoreMaxAvg(new int[] { 7 }, new int[] { 8 }), 0.001);
        assertTrue(boqa.resScoreMaxAvg(new int[] { 7 }, new int[] { 8 }) == boqa.resScoreMaxAvg(new int[] { 11 },
            new int[] { 8 }));

        /* Lin */
        assertEquals(0.243529, boqa.linScoreMaxAvg(new int[] { 7 }, new int[] { 8 }), 0.001);
        assertEquals(1, boqa.linScoreMaxAvg(new int[] { 12 }, new int[] { 12 }), 0.001);
        assertEquals(1, boqa.linScoreMaxAvg(new int[] { 0 }, new int[] { 0 }), 0.001);

        /* JC */
        assertEquals(0.42, boqa.jcScoreMaxAvg(new int[] { 7 }, new int[] { 8 }), 0.001);
        assertEquals(1, boqa.jcScoreMaxAvg(new int[] { 12 }, new int[] { 12 }), 0.001);
        assertEquals(1, boqa.jcScoreMaxAvg(new int[] { 0 }, new int[] { 0 }), 0.001);

        /* The follow to scores represent an example for avoiding similarity measure */

        /* The terms match the terms of the item */
        assertEquals(0.9163, boqa.resScoreVsItem(new int[] { 3, 10 }, 2), 0.001);

        /* The terms don't match the terms of the item */
        assertEquals(1.26286432, boqa.resScoreVsItem(new int[] { 9, 10 }, 2), 0.001);

        /* The terms match the terms of the item */
        assertEquals(1.26286432, boqa.resScoreVsItem(new int[] { 12, 9 }, 0), 0.001);

        /* Some other values */
        assertEquals(0.91629073, boqa.resScoreVsItem(new int[] { 10 }, 0), 0.001);
        assertEquals(1.14733979, boqa.resScoreVsItem(new int[] { 9, 10, 12 }, 0), 0.001);

        /* Now a bigger test */

        /* Item 2 */
        boolean[] obs = new boolean[boqa.getSlimGraph().getNumberOfVertices()];
        int item = 2;
        Observations o = new Observations();
        o.observations = obs;
        System.out.println("Testing item " + item);

        for (int i = 0; i < boqa.items2DirectTerms[item].length; i++) {
            int t = boqa.items2DirectTerms[item][i];
            obs[t] = true;
            boqa.activateAncestors(t, obs);
        }

        System.out.println("Observations");
        for (int i = 0; i < boqa.getSlimGraph().getNumberOfVertices(); i++) {
            if (obs[i]) {
                System.out.print(i + " ");
            }
        }
        System.out.println();

        Result resnikResult = boqa.resnikScore(obs, true, new Random(3));
        Result linResult = boqa.linScore(obs, true, new Random(3));
        Result jcResult = boqa.jcScore(obs, true, new Random(3));
        Result fabnResult = boqa.assignMarginals(o, false);

        System.out.println("Resnik");
        for (int i = 0; i < boqa.allItemList.size(); i++) {
            System.out.println(i + "  p=" + resnikResult.getMarginal(i) + " score=" + resnikResult.getScore(i));
        }

        System.out.println("Lin");
        for (int i = 0; i < boqa.allItemList.size(); i++) {
            System.out.println(i + "  p=" + linResult.getMarginal(i) + " score=" + linResult.getScore(i));
        }

        System.out.println("JC");
        for (int i = 0; i < boqa.allItemList.size(); i++) {
            System.out.println(i + "  p=" + jcResult.getMarginal(i) + " score=" + jcResult.getScore(i));
        }

        System.out.println("FABN");
        for (int i = 0; i < boqa.allItemList.size(); i++) {
            System.out.println(i + " marg=" + fabnResult.getMarginal(i) + " score=" + fabnResult.getScore(i));
        }

        double[] resnikP = new double[] { 0.64384, 0.843968, 0.15518, 0.844376, 0.68832 };
        double[] resnikScore =
            new double[] { 0.4581453659370775, 0.2231435513142097, 0.916290731874155, 0.2231435513142097,
            0.11157177565710485 };
        double[] linScore =
            new double[] { 0.28466172096330344, 0.2841853289422175, 1.0, 0.24352920263396993, 0.19583714006727046 };
        double[] linP = new double[] { 0.64384, 0.754716, 0.02238, 0.823132, 0.577072 };
        double[] jcScore =
            new double[] { 0.44323653311493805, 0.4576751569317141, 0.9999999999999998, 0.5552427170907795,
            0.6003949092992065 };
        double[] jcP = new double[] { 0.755548, 0.866124, 0.02238, 0.667388, 0.622036 };
        double[] fabnScores =
            new double[] { -0.9724346539489889, -1.3317577761951482, 2.2926928336673593, -0.6160410530454002,
            -0.6160410530454002 };
        double[] fabnMarginals =
            new double[] { 0.032533088008779756, 0.022712933995512156, 0.8518284456000037, 0.046462766197852126,
            0.046462766197852126 };

        for (int i = 0; i < boqa.allItemList.size(); i++) {
            System.out.println(i);
            assertEquals(resnikP[i], resnikResult.getMarginal(i), 0.05);
            assertEquals(resnikScore[i], resnikResult.getScore(i), 0.0001);
            assertEquals(linP[i], linResult.getMarginal(i), 0.05);
            assertEquals(linScore[i], linResult.getScore(i), 0.0001);
            assertEquals(jcP[i], jcResult.getMarginal(i), 0.05);
            assertEquals(jcScore[i], jcResult.getScore(i), 0.0001);
            assertEquals(fabnScores[i], fabnResult.getScore(i), 0.0001);
            assertEquals(fabnMarginals[i], fabnResult.getMarginal(i), 0.0001);
        }

        /* Item 3 */
        obs = new boolean[boqa.getSlimGraph().getNumberOfVertices()];
        item = 3;
        o = new Observations();
        o.observations = obs;

        System.out.println("Testing item " + item);

        for (int i = 0; i < boqa.items2DirectTerms[item].length; i++) {
            int t = boqa.items2DirectTerms[item][i];
            obs[t] = true;
            boqa.activateAncestors(t, obs);
        }

        resnikResult = boqa.resnikScore(obs, true, new Random(3));
        linResult = boqa.linScore(obs, true, new Random(3));
        jcResult = boqa.jcScore(obs, true, new Random(3));
        fabnResult = boqa.assignMarginals(o, false);

        System.out.println("Resnik");
        for (int i = 0; i < boqa.allItemList.size(); i++) {
            System.out.println(i + "  p=" + resnikResult.getMarginal(i) + " score=" + resnikResult.getScore(i));
        }

        System.out.println("Lin");
        for (int i = 0; i < boqa.allItemList.size(); i++) {
            System.out.println(i + "  p=" + linResult.getMarginal(i) + " score=" + linResult.getScore(i));
        }

        System.out.println("JC");
        for (int i = 0; i < boqa.allItemList.size(); i++) {
            System.out.println(i + "  p=" + jcResult.getMarginal(i) + " score=" + jcResult.getScore(i));
        }

        System.out.println("FABN");
        for (int i = 0; i < boqa.allItemList.size(); i++) {
            System.out.println(i + " marg=" + fabnResult.getMarginal(i) + " score=" + fabnResult.getScore(i));
        }

        resnikScore =
            new double[] { 0.11157177565710485, 0.2231435513142097, 0.2231435513142097, 0.916290731874155,
            0.11157177565710485 };
        resnikP = new double[] { 0.821472, 0.843968, 0.799752, 0.200844, 0.68832 };
        linScore =
            new double[] { 0.19583714006727046, 0.2841853289422175, 0.24352920263396993, 1.0, 0.08834818887494705 };
        linP = new double[] { 0.821472, 0.754716, 0.755148, 0.044964, 0.755296 };
        jcScore =
            new double[] { 0.43712271203780284, 0.4576751569317141, 0.5552427170907795, 0.9999999999999998,
            0.6003949092992065 };
        jcP = new double[] { 0.889184, 0.866124, 0.6225, 0.044964, 0.622036 };
        fabnScores =
            new double[] { -1.633587994564142, -1.633587994564142, -1.3091882336053455, 2.196264120323024,
            -1.3063742210362417 };
        fabnMarginals =
            new double[] { 0.019674959868555657, 0.019674959868555657, 0.027214407363261438, 0.9061445759999325,
            0.027291096899695524 };

        for (int i = 0; i < boqa.allItemList.size(); i++) {
            System.out.println(i);
            assertEquals(resnikP[i], resnikResult.getMarginal(i), 0.05);
            assertEquals(resnikScore[i], resnikResult.getScore(i), 0.0001);
            assertEquals(linP[i], linResult.getMarginal(i), 0.05);
            assertEquals(linScore[i], linResult.getScore(i), 0.0001);
            assertEquals(jcP[i], jcResult.getMarginal(i), 0.05);
            assertEquals(jcScore[i], jcResult.getScore(i), 0.0001);
            assertEquals(fabnScores[i], fabnResult.getScore(i), 0.0001);
            assertEquals(fabnMarginals[i], fabnResult.getMarginal(i), 0.0001);
        }
    }

    @Test
    @Ignore
    public void testBOQAOnInternalOntology() throws FileNotFoundException
    {
        final InternalDatafiles data = new InternalDatafiles();
        assertEquals(15, data.graph.getNumberOfTerms());

        final BOQA boqa = new BOQA();
        boqa.setConsiderFrequenciesOnly(false);
        boqa.setPrecalculateItemMaxs(true);
        boqa.setCacheScoreDistribution(true);
        boqa.setPrecalculateScoreDistribution(true);
        boqa.setStoreScoreDistriubtion(false);
        boqa.setTryLoadingScoreDistribution(false);
        boqa.setMaxQuerySizeForCachedDistribution(4);
        boqa.setup(data.graph, data.assoc);

        /* Write out the graph with ICs */
        data.getGraphWithItems().writeDOT(new FileOutputStream("full-with-ics.dot"),
            new DotAttributesProvider<String>()
            {
                @Override
                public String getDotNodeAttributes(String vt)
                {
                    if (vt.startsWith("C")) {
                        StringBuilder info = new StringBuilder();
                        for (Term t : data.graph) {
                            if (t.getName().equals(vt)) {
                                int idx = boqa.getSlimGraph().getVertexIndex(t);
                                info.append("\\n" + boqa.getNumberOfItemsAnnotatedToTerm(idx) + " "
                                    + String.format("%g", boqa.terms2IC[idx]));
                                break;
                            }
                        }

                        return "label=\"" + vt + info.toString() + "\"";

                    } else {
                        return "shape=\"box\",label=\"" + vt + "\"";
                    }
                }

                @Override
                public String getDotEdgeAttributes(String src, String dest)
                {
                    return "dir=\"back\"";
                }
            });

        System.out.println("Term Mapping");
        for (int i = 0; i < boqa.getSlimGraph().getNumberOfVertices(); i++) {
            System.out.println(i + " ->  " + boqa.getSlimGraph().getVertex(i).getIDAsString());
        }

        System.out.println("Item Mapping");
        for (int i = 0; i < boqa.allItemList.size(); i++) {
            System.out.println(i + " -> " + boqa.allItemList.get(i));
        }

        BOQA boqaNoPrecalc = new BOQA();
        boqaNoPrecalc.setConsiderFrequenciesOnly(false);
        boqaNoPrecalc.setPrecalculateItemMaxs(false);
        boqaNoPrecalc.setCacheScoreDistribution(false);
        boqaNoPrecalc.setPrecalculateScoreDistribution(false);
        boqaNoPrecalc.setStoreScoreDistriubtion(false);
        boqaNoPrecalc.setTryLoadingScoreDistribution(false);
        boqaNoPrecalc.setMaxQuerySizeForCachedDistribution(4);
        boqaNoPrecalc.setup(data.graph, data.assoc);

        checkInternalSimValues(boqa);
        checkInternalSimValues(boqaNoPrecalc);
    }

    @Test
    public void testBenchmarkOnInternalOntology() throws InterruptedException, IOException
    {
        final InternalDatafiles data = new InternalDatafiles();
        BOQA boqa = new BOQA();
        boqa.setConsiderFrequenciesOnly(false);
        boqa.setSizeOfScoreDistribution(1000);
        boqa.setTryLoadingScoreDistribution(false);
        boqa.setSimulationMaxTerms(3);
        boqa.setMaxQuerySizeForCachedDistribution(6);

        boqa.setup(data.graph, data.assoc);

        Benchmark benchmark = new Benchmark();
        benchmark.setResultBaseName("internal");
        benchmark.benchmark(boqa);
    }

    /**
     * A helper function to check similarity values for the HPO.
     *
     * @param boqa
     */
    private void checkHPOSimValues(final BOQA boqa)
    {
        assertEquals(0.006997929, boqa.resScoreMaxAvgVsItem(new int[] { 10, 12 }, 4), 0.000001);
        assertEquals(0.162568779, boqa.resScoreMaxAvgVsItem(new int[] { 101, 1222, 1300, 2011 }, 78), 0.000001);
    }

    @Test
    @Ignore
    public void testVsOldItemMax() throws InterruptedException, IOException
    {
        final BOQA boqa = new BOQA();

        Datafiles df = hpo;

        boqa.setConsiderFrequenciesOnly(false);
        boqa.setCacheScoreDistribution(false);
        boqa.setPrecalculateScoreDistribution(false);
        boqa.setStoreScoreDistriubtion(false);
        boqa.setPrecalculateItemMaxs(true);
        boqa.setup(df.graph, df.assoc);

        /* This is older code which we keep for testing here */
        int[][] micaForItem = new int[boqa.allItemList.size()][boqa.getSlimGraph().getNumberOfVertices()];
        for (int item = 0; item < boqa.allItemList.size(); item++) {
            /* The fixed set */
            int[] t2 = boqa.items2DirectTerms[item];

            for (int to = 0; to < boqa.getSlimGraph().getNumberOfVertices(); to++) {
                double maxIC = Double.NEGATIVE_INFINITY;
                int maxCommon = -1;

                for (int ti : t2) {
                    int common = boqa.getCommonAncestorWithMaxIC(to, ti);
                    if (boqa.terms2IC[common] > maxIC) {
                        maxIC = boqa.terms2IC[common];
                        maxCommon = common;
                    }
                }
                micaForItem[item][to] = maxCommon;
            }
        }

        /* Now the test */
        for (int i = 0; i < micaForItem.length; i++) {
            for (int j = 0; j < micaForItem[i].length; j++) {
                assertEquals(boqa.terms2IC[micaForItem[i][j]], boqa.resnikTermSim.maxScoreForItem[i][j], 0.00001);
            }
        }

        checkHPOSimValues(boqa);
    }

    @Test
    public void testMostSpecificOnHPO() throws InterruptedException, IOException
    {
        final BOQA boqa = new BOQA();

        boqa.setConsiderFrequenciesOnly(false);
        boqa.setCacheScoreDistribution(false);
        boqa.setPrecalculateScoreDistribution(false);
        boqa.setStoreScoreDistriubtion(false);
        boqa.setPrecalculateItemMaxs(false);
        boqa.setSimulationMaxTerms(-1);
        boqa.setup(hpo.graph, hpo.assoc);

        /* Check, mostSpecifc function */
        for (int i = 0; i < 5000; i++) {
            Random rnd = new Random(System.currentTimeMillis());

            int item = rnd.nextInt(boqa.allItemList.size());

            Observations obs = boqa.generateObservations(item, rnd);
            BOQA.IntArray sparse = new BOQA.IntArray(obs.observations);
            int[] mst = boqa.mostSpecificTerms(sparse.get());

            /* Get full observation according to mostSpecificTerms() */
            boolean[] actualObservations = new boolean[boqa.getSlimGraph().getNumberOfVertices()];
            for (int t : mst) {
                for (i = 0; i < boqa.term2Ancestors[t].length; i++) {
                    actualObservations[boqa.term2Ancestors[t][i]] = true;
                }
            }

            /* Get full observations according to source array */
            boolean[] expectedObservations = new boolean[boqa.getSlimGraph().getNumberOfVertices()];
            for (int t : sparse.get()) {
                for (i = 0; i < boqa.term2Ancestors[t].length; i++) {
                    expectedObservations[boqa.term2Ancestors[t][i]] = true;
                }
            }

            for (i = 0; i < actualObservations.length; i++) {
                assertEquals(expectedObservations[i], actualObservations[i]);
            }
        }
    }

    @Test
    public void testLargeNumberOfItems() throws IOException, OBOParserException, URISyntaxException
    {
        Random rnd = new Random(2);

        final BOQA boqa = new BOQA();

        OBOParser hpoParser = new OBOParser(
            new File(ClassLoader.getSystemResource("human-phenotype-ontology.obo.gz").toURI()).getCanonicalPath());
        hpoParser.doParse();
        TermContainer tc = new TermContainer(hpoParser.getTermMap(), hpoParser.getFormatVersion(), hpoParser.getDate());
        Ontology ontology = new Ontology(tc);
        SlimDirectedGraphView<Term> slim = ontology.getSlimGraphView();

        AssociationContainer assocs = new AssociationContainer();

        for (int i = 0; i < 10000; i++) {
            ByteString item = new ByteString("item" + i);

            // Association a = new Association(item,slim.getVertex(10).getIDAsString());
            // assocs.addAssociation(a);

            for (int j = 0; j < rnd.nextInt(16) + 2; j++) {
                Term t;
                do {
                    t = slim.getVertex(rnd.nextInt(slim.getNumberOfVertices()));
                } while (t.isObsolete());
                Association a = new Association(item, t.getIDAsString());
                assocs.addAssociation(a);
            }
        }

        this.logger.info("Constructed data set");
        boqa.setConsiderFrequenciesOnly(false);
        boqa.setPrecalculateScoreDistribution(false);
        boqa.setCacheScoreDistribution(false);
        boqa.setPrecalculateItemMaxs(false);
        boqa.setup(ontology, assocs);
        this.logger.info("Setted up ontology and associations");

        Observations o = new Observations();
        o.observations = new boolean[boqa.getOntology().getNumberOfTerms()];

        long start = System.nanoTime();
        this.logger.info("Calculating");
        boqa.assignMarginals(o, false, 1);
        long end = System.nanoTime();

        this.logger.info(((end - start) / 1000 / 1000) + "ms");
    }
}
