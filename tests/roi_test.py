import nose.tools

from sign_detection.model.RegionOfInterest import RegionOfInterest


def test_IouCompleteOverlapTest():
    roi_a = RegionOfInterest(1, 1, 100, 100, -1)
    roi_b = RegionOfInterest(1, 1, 100, 100, -1)
    iou = roi_a.intersection_over_union(roi_b)
    expected = 1
    nose.tools.assert_equal(iou,
                            expected,
                            "Wrong IoU. Expected: {}, actual: {}".format(expected, iou))


def test_IouPartialOverlapTest():
    roi_a = RegionOfInterest(1, 1, 100, 100, -1)
    roi_b = RegionOfInterest(50, 1, 150, 100, -1)
    iou = roi_a.intersection_over_union(roi_b)
    expected = 0.33333
    nose.tools.assert_almost_equal(iou,
                                   expected,
                                   msg="Wrong IoU. Expected: {}, actual: {}".format(expected, iou),
                                   delta=0.01)


def test_IouNoOverlapTest():
    roi_a = RegionOfInterest(1, 1, 100, 100, -1)
    roi_b = RegionOfInterest(200, 1, 300, 100, -1)
    iou = roi_a.intersection_over_union(roi_b)
    expected = 0
    nose.tools.assert_equal(iou,
                            expected,
                            "Wrong IoU. Expected: {}, actual: {}".format(expected, iou))
