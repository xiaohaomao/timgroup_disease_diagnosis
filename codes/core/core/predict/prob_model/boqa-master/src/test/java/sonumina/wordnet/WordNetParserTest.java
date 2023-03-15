package sonumina.wordnet;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ontologizer.association.Association;
import ontologizer.association.AssociationContainer;
import ontologizer.go.Ontology;
import ontologizer.go.Term;
import ontologizer.go.TermContainer;
import ontologizer.go.TermID;
import ontologizer.types.ByteString;
import sonumina.boqa.calculation.BOQA;
import sonumina.math.graph.AbstractGraph.DotAttributesProvider;
import sonumina.math.graph.SlimDirectedGraphView;

public class WordNetParserTest
{
    private Logger logger = LoggerFactory.getLogger(WordNetParserTest.class);

    /**
     * Download and unpack the word net stuff.
     *
     * @throws IOException
     * @throws InterruptedException
     */
    @Before
    public void setup() throws IOException, InterruptedException
    {
        File dest = new File("WordNet-3.0.tar.bz2");
        if (!dest.exists()) {
            int c;
            Process p = Runtime.getRuntime().exec("wget http://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.bz2");

            while ((c = p.getErrorStream().read()) != -1) {
                this.logger.error(String.valueOf(c));
            }

            if (p.waitFor() != 0) {
                throw new RuntimeException("Getting wordnet failed!");
            }
        }

        File wordnet = new File("WordNet-3.0/dict/data.noun");
        if (!wordnet.exists()) {
            int c;
            Process p = Runtime.getRuntime().exec("tar vxjf WordNet-3.0.tar.bz2");
            while ((c = p.getErrorStream().read()) != -1) {
                this.logger.error(String.valueOf(c));
            }

            if (p.waitFor() != 0) {
                throw new RuntimeException("Extracting wordnet failed!");
            }
        }
    }

    @Test
    public void testWordnetParser() throws IOException
    {
        TermContainer tc = WordNetParser.parserWordnet("WordNet-3.0/dict/data.noun");
        Ontology ontology = new Ontology(tc);

        Set<TermID> ts = new HashSet<TermID>();
        // ts.addAll(ontology.getTermsOfInducedGraph(null, ontology.getTerm("WNO:09571693").getID())); /* Orion */
        // ts.addAll(ontology.getTermsOfInducedGraph(null, ontology.getTerm("WNO:09380117").getID())); /* Orion */
        ts.addAll(ontology.getTermsOfInducedGraph(null, ontology.getTerm("WNO:09917593").getID())); /* Child */
        ts.addAll(ontology.getTermsOfInducedGraph(null, ontology.getTerm("WNO:05560787").getID())); /* Leg */

        ontology.getGraph().writeDOT(new FileOutputStream(new File("test.dot")),
            ontology.getSetOfTermsFromSetOfTermIds(ts), new DotAttributesProvider<Term>()
            {
                @Override
                public String getDotNodeAttributes(Term vt)
                {
                    return "label=\"" + vt.getName() + "\"";
                }
            });
    }

    @Test
    @Ignore("Getting [The ontology graph contains cycles] error...")
    public void testLargeNumberOfItems() throws IOException
    {
        Random rnd = new Random(2);

        TermContainer tc = WordNetParser.parserWordnet("WordNet-3.0/dict/data.noun");
        Ontology ontology = new Ontology(tc);
        SlimDirectedGraphView<Term> slim = ontology.getSlimGraphView();

        AssociationContainer assocs = new AssociationContainer();

        for (int i = 0; i < 100000; i++) {
            ByteString item = new ByteString("item" + i);

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
        final BOQA boqa = new BOQA();
        boqa.setup(ontology, assocs);
        this.logger.info("Setted up ontology and associations");

    }
}
