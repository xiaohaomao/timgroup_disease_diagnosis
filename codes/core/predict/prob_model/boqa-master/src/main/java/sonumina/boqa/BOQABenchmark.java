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

package sonumina.boqa;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

import ontologizer.statistics.None;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ontologizer.GlobalPreferences;
import ontologizer.OntologizerThreadGroups;
import ontologizer.benchmark.Datafiles;
import ontologizer.go.Term;
import ontologizer.types.ByteString;
import sonumina.boqa.calculation.BOQA;
import sonumina.boqa.calculation.Observations;

/**
 * Main entry point of the BOQA benchmark.
 *
 * @author Sebastian Bauer
 */
public class BOQABenchmark
{
    static private String ontologyPath;

    static private String annotationPath;

    static private String patientPath;

    static private String outPath;

    static private boolean useFreq = false;

    static private int topk = -1;

    static private BOQA boqa;

    static private HashMap<Integer, ByteString> omimMap = null;

    static private Logger logger = LoggerFactory.getLogger(BOQABenchmark.class);

    /**
     * Parses the command line and returns a corresponding BOQA object.
     *
     * @param args
     */
    public static void parseCommandLine(String[] args)
    {
        Options opt = new Options();
        opt.addOption("o", "ontology", true, "Path or URL to the ontology file.");
        opt.addOption("a", "annotations", true, "Path or URL to files containing annotations.");
        opt.addOption("p", "patient", true, "Path to directory with patients");
        opt.addOption("d", "out", true, "Path to output directory");
        opt.addOption("f", "freq", true, "Take frequency into account");    // HY
        opt.addOption("k", "topk", true, "Output top k diseases");    // HY
        opt.addOption("h", "help", false, "Shows this help");

        try {
            GnuParser parser = new GnuParser();
            CommandLine cl;
            cl = parser.parse(opt, args);

            if (cl.hasOption('h')) {
                HelpFormatter f = new HelpFormatter();
                f.printHelp(BOQABenchmark.class.getName(), opt);
                System.exit(0);
            }

            ontologyPath = cl.getOptionValue('o', ontologyPath);
            annotationPath = cl.getOptionValue('a', annotationPath);
            patientPath = cl.getOptionValue('p');
            outPath = cl.getOptionValue('d');
            if (cl.hasOption('f'))  // HY
                useFreq = cl.getOptionValue('f').equals("1");
            if (cl.hasOption('k'))  // HY
                topk = Integer.parseInt(cl.getOptionValue('k'));

        } catch (ParseException e) {
            logger.error("Failed to parse commandline: " + e.getLocalizedMessage());
            System.exit(1);
        }

    }

    public static void addTermAndAncestors(Term t, Observations obsv)
    {
        try {
            int id = boqa.getTermIndex(t);
            obsv.observations[id] = true;
            boqa.activateAncestors(id, obsv.observations);

        } catch (NullPointerException e) {
            logger.warn(String.valueOf(t));

            for (Term p : boqa.getOntology().getTermParents(t)) {
                logger.warn("Parent: " + p);
                addTermAndAncestors(p, obsv);
            }
        }
    }

    /**
     * @param hpoTermList alist of HPO terms separated by comma, e.g., "HP:0000407,HP:0009830,HP:0002858".
     */
    private static ArrayList<String> initializeHPOTermList(String hpoTermList)
    {
        String A[] = hpoTermList.split(",");
        ArrayList<String> hpoList = new ArrayList<String>();

        for (String a : A) {
            a = a.trim();

            /* A well formed HPO term starts with "HP:" and has ten characters. */
            if (!a.startsWith("HP:") || a.length() != 10) {
                logger.warn(String.format(
                    "Error: malformed HPO input string \"%s\". Could not parse term \"%s\"", hpoTermList, a));

            }

            hpoList.add(a);
        }

        return hpoList;
    }

    private static ArrayList<String> preformBOQACalculations(ArrayList<String> hpoList)
    {
        Observations o = new Observations();
        o.observations = new boolean[boqa.getOntology().getNumberOfTerms()];

        // Add all hpo terms with ancestors to array of booleans
        for (String hpo : hpoList) {
            Term t = boqa.getOntology().getTerm(hpo);
            addTermAndAncestors(t, o);
        }

        // Get marginals
        final BOQA.Result res = boqa.assignMarginals(o, useFreq, 1);    //HY

        // All of this is sorting diseases by marginals
        Integer[] order = new Integer[res.size()];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }

        Arrays.sort(order, new Comparator<Integer>()
        {
            @Override
            public int compare(Integer o1, Integer o2)
            {
                if (res.getMarginal(o1) < res.getMarginal(o2)) {
                    return 1;
                }
                if (res.getMarginal(o1) > res.getMarginal(o2)) {
                    return -1;
                }
                return 0;
            }
        });

        // Get top 20 results
        ArrayList<String> results = new ArrayList<String>();
//        for (int i = 0; i < 20; i++) {
        int resLen = topk == -1? order.length: topk;
        for (int i = 0; i < resLen; i++) {  // HY
            int id = order[i];

            results.add(res.getMarginal(id) + "\t" + new String(BOQABenchmark.omimMap.get(id).getBytes(), StandardCharsets.UTF_8));
        }

        return results;
    }

    /**
     * The main entry.
     *
     * @param args
     * @throws IOException
     * @throws InterruptedException
     */
    public static void main(String[] args) throws InterruptedException, IOException
    {
        parseCommandLine(args);

        BOQA boqa = new BOQA();
        boqa.setConsiderFrequenciesOnly(false);
        boqa.setPrecalculateScoreDistribution(false);
        boqa.setCacheScoreDistribution(false);
        boqa.setPrecalculateItemMaxs(false);
        boqa.setPrecalculateMaxICs(false);
        boqa.setMaxFrequencyTerms(2);
        boqa.setPrecalculateJaccard(false);

        BOQABenchmark.boqa = boqa;

        GlobalPreferences.setProxyPort(888);
        GlobalPreferences.setProxyHost("realproxy.charite.de");

        // Initialize boqa
        Datafiles df = new Datafiles(ontologyPath, annotationPath);
        boqa.setup(df.graph, df.assoc);

        // Set up our index -> OMIM mapping by flipping the OMIM -> Index mapping in boqa
        Set<Map.Entry<ByteString, Integer>> omimtonum = boqa.item2Index.entrySet();
        omimMap = new HashMap<Integer, ByteString>(omimtonum.size());

        for (Map.Entry<ByteString, Integer> item : omimtonum) {
            omimMap.put(item.getValue(), item.getKey());
        }

        // HY

        ArrayList< ArrayList<String> > hpoLists = new ArrayList< ArrayList<String> >();
        // read data
        Scanner s = new Scanner(new File(patientPath));
        while (s.hasNext()){
            String hpoTerms = s.nextLine();
            // Get seperated terms
            ArrayList<String> hpoTermList = BOQABenchmark.initializeHPOTermList(hpoTerms);
            hpoLists.add(hpoTermList);
        }
        // predict
        ArrayList<String> results = new ArrayList<String>();
        for (ArrayList<String> hpoList: hpoLists){
            results.addAll(BOQABenchmark.preformBOQACalculations(hpoList));
            results.add("====");
        }
        Files.write(Paths.get(outPath), results, StandardCharsets.UTF_8);


//        // Read in hpo files
//        Charset utf8 = StandardCharsets.UTF_8;
//        File inFolder = new File(patientPath);
//        String[] files = inFolder.list();
//        Scanner s;
//        for (String f : files) {
//
//            if (f.endsWith("_hpo.txt")) {
//                s = new Scanner(new File(patientPath + File.separator + f));
//                String hpoTerms = s.nextLine();
//
//                // Get seperated terms
//                ArrayList<String> hpoTermList = BOQABenchmark.initializeHPOTermList(hpoTerms);
//
//                // Do actual calculations
//                ArrayList<String> data = BOQABenchmark.preformBOQACalculations(hpoTermList);
//                Files.write(Paths.get(outPath + File.separator + f + ".results"), data, utf8);
//            }
//        }

//        ArrayList<String> test =
//            BOQABenchmark.initializeHPOTermList("HP:0000163,HP:0002015,HP:0006292,HP:0000234,HP:0000585,HP:0000276");
//        BOQABenchmark.preformBOQACalculations(test);

        OntologizerThreadGroups.workerThreadGroup.interrupt();
    }
}
