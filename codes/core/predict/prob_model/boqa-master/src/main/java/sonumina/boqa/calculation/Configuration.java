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
 * Class to count the different cases.
 *
 * @author Sebastian Bauer
 */
final public class Configuration implements Cloneable
{
    public static enum NodeCase
    {
        FAULT,
        TRUE_POSITIVE,
        FALSE_POSITIVE,
        TRUE_NEGATIVE,
        FALSE_NEGATIVE,
        INHERIT_TRUE,
        INHERIT_FALSE
    }

    final private int[] stats = new int[NodeCase.values().length];

    final public void increment(NodeCase c)
    {
        this.stats[c.ordinal()]++;
    }

    final public void decrement(NodeCase c)
    {
        this.stats[c.ordinal()]--;
    }

    @Override
    public String toString()
    {
        String str = "";
        for (int i = 0; i < this.stats.length; i++) {
            str += " " + NodeCase.values()[i].name() + ": " + this.stats[i] + "\n";
        }

        return str;
    }

    /**
     * Get the number of observed cases for the given case.
     *
     * @param c
     * @return
     */
    final public int getCases(NodeCase c)
    {
        return this.stats[c.ordinal()];
    }

    /**
     * Returns the total number of cases that were tracked.
     *
     * @return
     */
    final public int getTotalCases()
    {
        int c = 0;
        for (int stat : this.stats) {
            c += stat;
        }
        return c;
    }

    /**
     * Returns the false positive rate.
     *
     * @return
     */
    final public double falsePositiveRate()
    {
        return getCases(Configuration.NodeCase.FALSE_POSITIVE)
            / (double) (getCases(Configuration.NodeCase.FALSE_POSITIVE)
                + getCases(Configuration.NodeCase.TRUE_NEGATIVE));
    }

    /**
     * Return false negative rate.
     *
     * @return
     */
    final public double falseNegativeRate()
    {
        return getCases(Configuration.NodeCase.FALSE_NEGATIVE)
            / (double) (getCases(Configuration.NodeCase.FALSE_NEGATIVE)
                + getCases(Configuration.NodeCase.TRUE_POSITIVE));
    }

    /**
     * Returns the log score of the summarized configuration.
     *
     * @param alpha
     * @param beta
     * @return
     */
    final public double getScore(double alpha, double beta)
    {
        return Math.log(beta) * getCases(NodeCase.FALSE_NEGATIVE) +
            Math.log(alpha) * getCases(NodeCase.FALSE_POSITIVE) +
            Math.log(1 - beta) * getCases(NodeCase.TRUE_POSITIVE) +
            Math.log(1 - alpha) * getCases(NodeCase.TRUE_NEGATIVE) +
            Math.log(1) * getCases(NodeCase.INHERIT_FALSE) + /* 0 */
            Math.log(1) * getCases(NodeCase.INHERIT_TRUE); /* 0 */
    }

    /**
     * Adds the given stat to this one.
     *
     * @param toAdd
     */
    final public void add(Configuration toAdd)
    {
        for (int i = 0; i < this.stats.length; i++) {
            this.stats[i] += toAdd.stats[i];
        }
    }

    /**
     * Clear the stats.
     */
    final public void clear()
    {
        for (int i = 0; i < this.stats.length; i++) {
            this.stats[i] = 0;
        }
    }

    @Override
    final public Configuration clone()
    {
        Configuration c = new Configuration();
        for (int i = 0; i < this.stats.length; i++) {
            c.stats[i] = this.stats[i];
        }
        return c;
    }

    public boolean equals(Configuration obj)
    {
        for (int i = 0; i < obj.stats.length; i++) {
            if (obj.stats[i] != this.stats[i]) {
                return false;
            }
        }
        return true;
    }
}
