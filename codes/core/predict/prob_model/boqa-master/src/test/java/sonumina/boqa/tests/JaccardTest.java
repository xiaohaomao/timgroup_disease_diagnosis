package sonumina.boqa.tests;

import org.junit.Assert;
import org.junit.Test;

import sonumina.boqa.calculation.BOQA;

public class JaccardTest
{
    private void testJaccardWithBoqa(BOQA boqa)
    {
        Assert.assertEquals(1.0, boqa.jaccard(0, 0), 0.0001);
        Assert.assertEquals(1.0, boqa.jaccard(0, 1), 0.0001);
        Assert.assertEquals(1.0, boqa.jaccard(1, 1), 0.0001);
        Assert.assertEquals(0.5, boqa.jaccard(9, 12), 0.0001);
        Assert.assertEquals(0.0, boqa.jaccard(9, 10), 0.0001);
        Assert.assertEquals(0.25, boqa.jaccard(6, 14), 0.0001);
        Assert.assertEquals(0.25, boqa.jaccard(14, 6), 0.0001);
        Assert.assertEquals(0.0, boqa.jaccard(10, 11), 0.0001);
        Assert.assertEquals(0.6, boqa.jaccard(3, 4), 0.0001);
        Assert.assertEquals(0.6, boqa.jaccard(4, 3), 0.0001);
        Assert.assertEquals(0.25, boqa.jaccard(10, 3), 0.0001);
        Assert.assertEquals(0.25, boqa.jaccard(3, 10), 0.0001);

        Assert.assertEquals(0.25, boqa.jaccard(3, 11), 0.0001);
        Assert.assertEquals(0.2, boqa.jaccard(3, 12), 0.0001);
        Assert.assertEquals(0.5, boqa.jaccard(3, 13), 0.0001);

        Assert.assertEquals(0.0, boqa.jaccard(10, 11), 0.0001);
        Assert.assertEquals(0.0, boqa.jaccard(10, 12), 0.0001);
        Assert.assertEquals(0.0, boqa.jaccard(10, 13), 0.0001);
    }

    @Test
    public void testJaccard()
    {
        InternalDatafiles data = new InternalDatafiles();
        BOQA boqa = new BOQA();
        boqa.setConsiderFrequenciesOnly(false);
        boqa.setPrecalculateScoreDistribution(false);
        boqa.setup(data.graph, data.assoc);
        testJaccardWithBoqa(boqa);
    }

    @Test
    public void testJaccardPrecalculated()
    {
        InternalDatafiles data = new InternalDatafiles();
        BOQA boqa = new BOQA();
        boqa.setConsiderFrequenciesOnly(false);
        boqa.setPrecalculateScoreDistribution(false);
        boqa.setPrecalculateJaccard(true);
        boqa.setup(data.graph, data.assoc);
        testJaccardWithBoqa(boqa);
    }

}
