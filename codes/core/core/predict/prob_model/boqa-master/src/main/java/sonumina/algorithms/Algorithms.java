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

package sonumina.algorithms;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class Algorithms
{
    public static interface IVertexDistance<V>
    {
        /**
         * Returns the distance between vertex a and vertex b.
         *
         * @param a
         * @param b
         * @return
         */
        public double distance(V a, V b);
    }

    /**
     * An heuristics to solve the TSP.
     *
     * @param vertices
     * @param start
     * @param distance
     * @return
     */
    @SuppressWarnings("unchecked")
    public static <V> List<V> approximatedTSP(Collection<V> vertices, V start, IVertexDistance<V> distance)
    {
        Object[] toDo = new Object[vertices.size() - 1];
        int toDoLength = toDo.length;
        ArrayList<V> list = new ArrayList<V>();
        list.add(start);

        int i = 0;

        for (V v : vertices) {
            if (v != start) {
                toDo[i++] = v;
            }
        }

        while (toDoLength > 0) {
            double minDistance = Double.MAX_VALUE;
            int newStartIndex = -1;

            for (i = 0; i < toDoLength; i++) {
                V v = (V) toDo[i];
                double nd = distance.distance(start, v);
                if (nd < minDistance) {
                    newStartIndex = i;
                    minDistance = nd;
                }
            }

            list.add((V) toDo[newStartIndex]);
            toDoLength--;
            if (toDoLength > 0) {
                toDo[newStartIndex] = toDo[toDoLength];
            }
        }

        return list;
    }

    /**
     * Returns a spare representation of the given vector. An element refers to an index that is true.
     *
     * @param dense
     * @return
     */
    public static int[] spareInt(boolean[] dense)
    {
        int c = 0;
        for (boolean element : dense) {
            if (element) {
                c++;
            }
        }

        int[] array = new int[c];
        c = 0;
        for (int i = 0; i < dense.length; i++) {
            if (dense[i]) {
                array[c++] = i;
            }
        }
        return array;
    }

    /**
     * Calculates the hamming distance of the sparsely represented vectors. Elements are assumed to be sorted.
     *
     * @return
     */
    public static int hammingDistanceSparse(int[] va, int[] vb)
    {
        int distance = 0;
        int i = 0, j = 0;

        while (i < va.length && j < vb.length) {
            if (va[i] < vb[j]) {
                distance++;
                i++;
            } else if (va[i] > vb[j]) {
                distance++;
                j++;
            } else {
                i++;
                j++;
            }
        }

        distance += va.length - i;
        distance += vb.length - j;
        return distance;
    }
}
