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

package sonumina.boqa.tests;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.HashSet;

import ontologizer.association.Association;
import ontologizer.association.AssociationContainer;
import ontologizer.association.Gene2Associations;
import ontologizer.benchmark.Datafiles;
import ontologizer.dotwriter.AbstractDotAttributesProvider;
import ontologizer.dotwriter.GODOTWriter;
import ontologizer.go.Ontology;
import ontologizer.go.ParentTermID;
import ontologizer.go.Term;
import ontologizer.go.TermContainer;
import ontologizer.go.TermID;
import ontologizer.go.TermRelation;
import ontologizer.types.ByteString;
import sonumina.math.graph.AbstractGraph.DotAttributesProvider;
import sonumina.math.graph.DirectedGraph;
import sonumina.math.graph.Edge;

/**
 * Class representing some internal data files.
 *
 * @author Sebastian Bauer
 */
public class InternalDatafiles extends Datafiles
{
    private DirectedGraph<String> graphWithItems;

    @SuppressWarnings("deprecation")
    public InternalDatafiles()
    {
        /* Go Graph */
        HashSet<Term> terms = new HashSet<Term>();
        Term c1 = new Term("GO:0000001", "C1");
        Term c2 = new Term("GO:0000002", "C2", new ParentTermID(c1.getID(), TermRelation.IS_A));
        Term c3 = new Term("GO:0000003", "C3", new ParentTermID(c1.getID(), TermRelation.IS_A));
        Term c4 = new Term("GO:0000004", "C4", new ParentTermID(c2.getID(), TermRelation.IS_A));
        Term c5 = new Term("GO:0000005", "C5", new ParentTermID(c2.getID(), TermRelation.IS_A));
        Term c6 =
            new Term("GO:0000006", "C6", new ParentTermID(c3.getID(), TermRelation.IS_A), new ParentTermID(c2.getID(),
                TermRelation.IS_A));
        Term c7 =
            new Term("GO:0000007", "C7", new ParentTermID(c5.getID(), TermRelation.IS_A), new ParentTermID(c6.getID(),
                TermRelation.IS_A));
        Term c8 = new Term("GO:0000008", "C8", new ParentTermID(c7.getID(), TermRelation.IS_A));
        Term c9 = new Term("GO:0000009", "C9", new ParentTermID(c7.getID(), TermRelation.IS_A));
        Term c10 = new Term("GO:0000010", "C10", new ParentTermID(c9.getID(), TermRelation.IS_A));
        Term c11 = new Term("GO:0000011", "C11", new ParentTermID(c9.getID(), TermRelation.IS_A));
        Term c12 = new Term("GO:0000012", "C12", new ParentTermID(c8.getID(), TermRelation.IS_A));
        Term c13 = new Term("GO:0000013", "C13", new ParentTermID(c8.getID(), TermRelation.IS_A));
        Term c14 = new Term("GO:0000014", "C14", new ParentTermID(c4.getID(), TermRelation.IS_A));
        Term c15 = new Term("GO:0000015", "C15", new ParentTermID(c4.getID(), TermRelation.IS_A));

        terms.add(c1);
        terms.add(c2);
        terms.add(c3);
        terms.add(c4);
        terms.add(c5);
        terms.add(c6);
        terms.add(c7);
        terms.add(c8);
        terms.add(c9);
        terms.add(c10);
        terms.add(c11);
        terms.add(c12);
        terms.add(c13);
        terms.add(c14);
        terms.add(c15);
        TermContainer termContainer = new TermContainer(terms, "", "");

        this.graph = new Ontology(termContainer);

        HashSet<TermID> tids = new HashSet<TermID>();
        for (Term term : terms) {
            tids.add(term.getID());
        }

        /* Associations */
        this.assoc = new AssociationContainer();

        this.assoc.addAssociation(new Association(new ByteString("item1"), 4));
        this.assoc.addAssociation(new Association(new ByteString("item1"), 11));

        this.assoc.addAssociation(new Association(new ByteString("item2"), 10));
        this.assoc.addAssociation(new Association(new ByteString("item2"), 13));

        this.assoc.addAssociation(new Association(new ByteString("item3"), 7));
        this.assoc.addAssociation(new Association(new ByteString("item3"), 15));

        this.assoc.addAssociation(new Association(new ByteString("item4"), 12));
        this.assoc.addAssociation(new Association(new ByteString("item4"), 13));
        this.assoc.addAssociation(new Association(new ByteString("item4"), 14));

        this.assoc.addAssociation(new Association(new ByteString("item5"), 6));
        this.assoc.addAssociation(new Association(new ByteString("item5"), 14));

        GODOTWriter.writeDOT(this.graph, new File("example.dot"), null, tids, new AbstractDotAttributesProvider()
        {
            @Override
            public String getDotNodeAttributes(TermID id)
            {

                return "label=\"" + InternalDatafiles.this.graph.getTerm(id).getName() + "\"";
            }
        });

        this.graphWithItems = new DirectedGraph<String>();
        for (Term term : terms) {
            this.graphWithItems.addVertex(term.getName());
        }

        for (Term term : terms) {
            for (ParentTermID pid : term.getParents()) {
                this.graphWithItems.addEdge(new Edge<String>(this.graph.getTerm(pid.termid).getName(), term.getName()));
            }
        }

        this.graphWithItems.addVertex("item1");
        this.graphWithItems.addVertex("item2");
        this.graphWithItems.addVertex("item3");
        this.graphWithItems.addVertex("item4");
        this.graphWithItems.addVertex("item5");

        for (Gene2Associations g2a : this.assoc) {
            for (TermID tid : g2a.getAssociations()) {
                this.graphWithItems.addEdge(new Edge<String>(this.graph.getTerm(tid).getName(), g2a.name().toString()));
            }
        }

        try {
            this.graphWithItems.writeDOT(new FileOutputStream("full.dot"), new DotAttributesProvider<String>()
            {
                @Override
                public String getDotNodeAttributes(String vt)
                {
                    if (vt.startsWith("C")) {
                        return "label=\"" + vt + "\"";
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
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    /**
     * Returns the graph together with items
     */
    public DirectedGraph<String> getGraphWithItems()
    {
        return this.graphWithItems;
    }

}
