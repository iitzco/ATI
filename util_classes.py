
class TrackingParameters():
    def __init__(self, hsv_tracking, full_tracking, occlusion_tracking):
        self.hsv_tracking = hsv_tracking
        self.full_tracking = full_tracking
        self.occlusion_tracking = occlusion_tracking


class TrackingContainer:
    def __init__(self, lin, lout, phi, mean, nmax, probability, tracking_parameters):
        self.lin = lin
        self.lout = lout
        self.phi = phi
        self.mean = mean
        self.nmax = nmax
        self.probability = probability
        self.hsv_tracking = tracking_parameters.hsv_tracking
        self.full_tracking = tracking_parameters.full_tracking
        self.occlusion_tracking = tracking_parameters.occlusion_tracking

        self.frame = 0
        self.max_area = -1

    def reset(self):
        self.frame = 0
        self.max_area = -1
