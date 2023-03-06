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

package sonumina.math.distribution;

import java.io.Serializable;
import java.util.Arrays;

/**
 * A simple class representing an approximated empirical probability. The distribution is approximated using equidistant
 * bins.
 *
 * @author Sebastian Bauer
 */
public class ApproximatedEmpiricalDistribution implements IDistribution, Serializable
{
    private static final long serialVersionUID = 1L;

    private double min;

    private double max;

    private int numberOfBins;

    private int[] cumCounts;

    public ApproximatedEmpiricalDistribution(double[] newObservations, int newNumberOfBins)
    {
        double[] observations = new double[newObservations.length];
        for (int i = 0; i < observations.length; i++) {
            observations[i] = newObservations[i];
        }
        Arrays.sort(observations);

        this.min = observations[0];
        this.max = observations[observations.length - 1];
        this.numberOfBins = newNumberOfBins;

        int[] counts = new int[this.numberOfBins];
        for (double observation : observations) {
            int bin = findBin(observation);
            if (bin < 0) {
                bin = 0;
            } else if (bin >= this.numberOfBins) {
                bin = this.numberOfBins - 1;
            }
            counts[bin]++;
        }

        for (int i = 1; i < this.numberOfBins; i++) {
            counts[i] += counts[i - 1];
        }

        this.cumCounts = counts;
    }

    private int findBin(double observation)
    {
        double binDbl = (observation - this.min) / (this.max - this.min) * this.numberOfBins;
        int bin = (int) Math.floor(binDbl);
        return bin;
    }

    @Override
    public double cdf(double x, boolean lowerTail)
    {
        int bin = findBin(x);
        if (bin < 0) {
            return 0;
        }
        if (bin >= this.numberOfBins) {
            return 1;
        }
        double cdf = this.cumCounts[bin] / (double) this.cumCounts[this.cumCounts.length - 1];
        return cdf;
    }

    /**
     * Returns for x P(X = x) which is not necessarily 0 be this is a discrete distribution.
     *
     * @param x
     * @return
     */
    public double prob(double x)
    {
        int bin = findBin(x);
        if (bin <= 0) {
            return cdf(x, false);
        }
        if (bin >= this.numberOfBins) {
            bin = this.numberOfBins - 1;
        }

        double p = (this.cumCounts[bin] - this.cumCounts[bin - 1]) / (double) this.cumCounts[this.cumCounts.length - 1];
        return p;
    }

    /**
     * Returns the maximum value.
     *
     * @return
     */
    public double getMax()
    {
        return this.max;
    }

    /**
     * Returns the minimum value.
     *
     * @return
     */
    public double getMin()
    {
        return this.min;
    }

    @Override
    public String toString()
    {
        StringBuilder str = new StringBuilder();

        str.append("min=" + this.min);
        str.append(" max=" + this.max + " ");

        for (int bin = 0; bin < this.numberOfBins; bin++) {
            double obs = bin * (this.max - this.min) / this.numberOfBins + this.min;
            str.append(obs);
            str.append(" (");
            str.append(this.cumCounts[bin]);
            str.append(") ");
        }
        return str.toString();
    };
}
