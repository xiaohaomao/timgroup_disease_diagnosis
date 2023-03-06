package sonumina.algorithms;

import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class AlgorithmsTest
{
    class Vector
    {
        String name;

        boolean[] v = new boolean[100];

        public Vector(String name, int[] ons)
        {
            this.name = name;
            for (int i = 0; i < ons.length; i++) {
                this.v[ons[i]] = true;
            }
        }
    }

    @Test
    public void testTSP()
    {
        Vector v1 = new Vector("v1", new int[] { 2, 6, 10, 23, 27, 30, 40, 76, 99 });
        Vector v2 = new Vector("v2", new int[] { 2, 6, 10, 23 });
        Vector v3 = new Vector("v3", new int[] { 2, 6, 10, 24 });
        Vector v4 = new Vector("v4", new int[] { 2, 6, 10 });
        Vector v5 = new Vector("v5", new int[] { 2, 6, 13, 23, 27, 30, 40, 76, 99 });
        Vector v6 = new Vector("v6", new int[] { 2, 4, 14, 22, 24, 34, 41, 45, 84 });

        LinkedList<Vector> vectors = new LinkedList<Vector>();
        vectors.add(v1);
        vectors.add(v2);
        vectors.add(v3);
        vectors.add(v4);
        vectors.add(v5);
        vectors.add(v6);

        List<Vector> vs = Algorithms.approximatedTSP(vectors, v4, new Algorithms.IVertexDistance<Vector>()
        {
            @Override
            public double distance(Vector a, Vector b)
            {
                int d = 0;
                for (int i = 0; i < a.v.length; i++) {
                    if (a.v[i] != b.v[i]) {
                        d++;
                    }
                }
                return d;
            };
        });

        assertEquals(6, vs.size());
        assertNotNull(vs);
    }

    @Test
    public void testHammingDistance()
    {
        Random rnd = new Random();

        boolean[] states1 = new boolean[10000];
        for (int i = 0; i < states1.length; i++) {
            states1[i] = rnd.nextInt(5) < 2;
        }

        boolean[] states2 = new boolean[10000];
        for (int i = 0; i < states2.length; i++) {
            states2[i] = rnd.nextInt(5) < 2;
        }

        int[] sparse1 = Algorithms.spareInt(states1);
        int[] sparse2 = Algorithms.spareInt(states2);

        int dist1 = 0;
        for (int i = 0; i < states1.length; i++) {
            if (states1[i] != states2[i]) {
                dist1++;
            }
        }

        assertEquals(dist1, Algorithms.hammingDistanceSparse(sparse1, sparse2));
        assertEquals(0, Algorithms.hammingDistanceSparse(new int[] {}, new int[] {}));
        assertEquals(2, Algorithms.hammingDistanceSparse(new int[] { 1 }, new int[] { 2 }));
    }
}
