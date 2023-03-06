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

package sonumina.boqa.calculation;

/**
 * A class to generate stepwise subsets with cardinality not greater than m of the set {0,1,...,n-1}. Note that an empty
 * subset is generated as well.
 *
 * @author sba
 */
class SubsetGenerator
{
    static public class Subset
    {
        /** Subset */
        public int[] j;

        /** Size of the subset */
        public int r;
    }

    private Subset subset;

    private int n;

    private int m;

    /** Indicates whether first subset has already been generated */
    private boolean firstSubset;

    /**
     * Constructor.
     *
     * @param n defines size of the set
     * @param m defines the maximum cardinality of the generated subsets.
     */
    public SubsetGenerator(int n, int m)
    {
        this.n = n;
        this.m = m;
        this.firstSubset = true;
        this.subset = new Subset();
    }

    /**
     * Returns the next subset or null if all subsets have already been created. Note that the returned array is read
     * only!
     *
     * @return
     */
    public Subset next()
    {
        if (this.subset.r == 0) {
            if (this.firstSubset) {
                this.firstSubset = false;
                return this.subset;
            }

            /* Special case when subset of an empty set or no subset should be generated */
            if (this.n == 0 || this.m == 0) {
                this.firstSubset = true;
                return null;
            }

            /* First call of next inside a subset generating phase */
            this.subset.j = new int[this.m];
            this.subset.r = 1;
            return this.subset;
        }

        int[] j = this.subset.j;
        int r = this.subset.r;

        if (j[r - 1] < this.n - 1 && r < this.m) {
            /* extend */
            j[r] = j[r - 1] + 1;
            r++;
        } else {
            /* modified reduce */
            if (j[r - 1] >= this.n - 1) {
                r--;
            }

            if (r == 0) {
                this.subset.r = 0;
                this.firstSubset = true;
                return null;
            }
            j[r - 1] = j[r - 1] + 1;
        }

        this.subset.r = r;
        return this.subset;
    }
}
