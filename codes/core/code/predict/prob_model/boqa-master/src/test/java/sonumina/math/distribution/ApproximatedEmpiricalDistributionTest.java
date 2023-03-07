package sonumina.math.distribution;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * A simple class representing an approximated empirical probability.
 *
 * @author Sebastian Bauer
 */
public class ApproximatedEmpiricalDistributionTest
{
    @Test
    public void test()
    {
        double[] obs = new double[100];
        for (int i = 0; i < obs.length; i++) {
            obs[i] = i;
        }

        ApproximatedEmpiricalDistribution dist = new ApproximatedEmpiricalDistribution(obs, 10);

        assertEquals(0.6, dist.cdf(51, false), 0.001);
        assertEquals(0.2, dist.cdf(11, false), 0.001);
        assertEquals(0.1, dist.cdf(0, false), 0.001);
        assertEquals(0, dist.cdf(-11, false), 0.001);
        assertEquals(1, dist.cdf(100, false), 0.001);
    }

    @Test
    public void test2()
    {
        double[] obs = new double[] {
        0.1, 0.11, 0.13, 0.231, 0.1213,
        0.1345, 0.33, 1.323, 2.32, 1.234,
        2.2, 1.1, 1.1, 1.65, 1.7,
        1.8, 1.9, 2.0, 2.1, 2.2
        };
        ApproximatedEmpiricalDistribution dist = new ApproximatedEmpiricalDistribution(obs, 1000);

        assertEquals(1, dist.cdf(3, false), 0.001);
        assertEquals(0.05, dist.cdf(0.1, false), 0.001);
        assertEquals(0.8, dist.cdf(2, false), 0.001);
    }
}
