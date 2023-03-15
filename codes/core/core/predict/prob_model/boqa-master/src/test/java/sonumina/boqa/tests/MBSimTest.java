package sonumina.boqa.tests;

import org.junit.Assert;
import org.junit.Test;

import sonumina.boqa.calculation.BOQA;

public class MBSimTest
{
    @Test
    public void mbSimTerm()
    {
        InternalDatafiles data = new InternalDatafiles();
        BOQA boqa = new BOQA();
        boqa.setConsiderFrequenciesOnly(false);
        boqa.setPrecalculateScoreDistribution(false);
        boqa.setup(data.graph, data.assoc);

        for (int i = 0; i < boqa.getSlimGraph().getNumberOfVertices(); i++) {
            System.out.println(i + " " + boqa.getNumberOfItemsAnnotatedToTerm(i) + " " + boqa.getIC(i));
        }

        Assert.assertEquals(0, boqa.mbTermSim(0, 0), 0.000001);
        Assert.assertEquals(0, boqa.mbTermSim(0, 1), 0.000001);
        Assert.assertEquals(0, boqa.mbTermSim(1, 0), 0.000001);
        Assert.assertEquals(0, boqa.mbTermSim(10, 11), 0.000001);
        Assert.assertEquals(0, boqa.mbTermSim(11, 10), 0.000001);
        Assert.assertEquals(0.6 * (-Math.log(4. / 5) - Math.log(4. / 5)) / 2, boqa.mbTermSim(3, 4), 0.0001);
        Assert.assertEquals(0.5 * (-Math.log(1. / 5) - Math.log(2. / 5)) / 2, boqa.mbTermSim(9, 12), 0.0001);
        Assert.assertEquals(0.5 * (-Math.log(1. / 5) - Math.log(2. / 5)) / 2, boqa.mbTermSim(12, 9), 0.0001);
        Assert.assertEquals(0.25 * (-Math.log(4. / 5) - Math.log(1. / 5)) / 2, boqa.mbTermSim(3, 10), 0.0001);
        Assert.assertEquals(0.6 * (-Math.log(4. / 5) - Math.log(4. / 5)) / 2,
            boqa.mbsim(new int[] { 3 }, new int[] { 4 }), 0.0001);
        Assert.assertEquals(0.5 * (-Math.log(1. / 5) - Math.log(2. / 5)) / 2,
            boqa.mbsim(new int[] { 9 }, new int[] { 12 }), 0.0001);
        Assert.assertEquals(0.5 * (-Math.log(1. / 5) - Math.log(2. / 5)) / 2,
            boqa.mbsim(new int[] { 12 }, new int[] { 9 }), 0.0001);

        /* Larger test */

        Assert.assertEquals(0.25 * (-Math.log(4. / 5) - Math.log(1. / 5)) / 2, boqa.mbTermSim(3, 11), 0.0001);
        Assert.assertEquals(0.2 * (-Math.log(4. / 5) - Math.log(2. / 5)) / 2, boqa.mbTermSim(3, 12), 0.0001);
        Assert.assertEquals(0.5 * (-Math.log(4. / 5) - Math.log(2. / 5)) / 2, boqa.mbTermSim(3, 13), 0.0001);
        /* Maximum of the three previous ones */
        Assert.assertEquals(0.5 * (-Math.log(4. / 5) - Math.log(2. / 5)) / 2, boqa.msim(3, new int[] { 11, 12, 13 }),
            0.0001);

        Assert.assertEquals(0.0, boqa.mbTermSim(10, 11), 0.0001);
        Assert.assertEquals(0.0, boqa.mbTermSim(10, 12), 0.0001);
        Assert.assertEquals(0.0, boqa.mbTermSim(10, 13), 0.0001);
        /* Maximum of the three previous ones */
        Assert.assertEquals(0, boqa.msim(10, new int[] { 11, 12, 13 }), 0.0001);

        /* 0.1424292853985456 */
        Assert.assertEquals(0.5 * 0.5 * (-Math.log(4. / 5) - Math.log(2. / 5)) / 2,
            boqa.mbsimUnsym(new int[] { 3, 10 }, new int[] { 11, 12, 13 }), 0.0001);

        /* The other part */
        double r1 = 0.25 * (-Math.log(4. / 5) - Math.log(1. / 5)) / 2;
        double r2 = 0.2 * (-Math.log(4. / 5) - Math.log(2. / 5)) / 2;
        double r3 = 0.5 * (-Math.log(4. / 5) - Math.log(2. / 5)) / 2;
        Assert.assertEquals(r1, boqa.msim(11, new int[] { 3, 10 }), 0.0001);
        Assert.assertEquals(r2, boqa.msim(12, new int[] { 3, 10 }), 0.0001);
        Assert.assertEquals(r3, boqa.msim(13, new int[] { 3, 10 }), 0.0001);
        Assert.assertEquals((r1 + r2 + r3) / 3, boqa.mbsimUnsym(new int[] { 11, 12, 13 }, new int[] { 3, 10 }), 0.0001);

        Assert.assertEquals((0.1424292853985456 + 0.20929156069482216) / 2,
            boqa.mbsim(new int[] { 11, 12, 13 }, new int[] { 3, 10 }), 0.0001);
    }
}
