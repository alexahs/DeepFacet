import os, sys
sys.path.append(os.path.abspath("../composite_systems"))
sys.path.append(os.path.abspath("../nets"))
sys.path.append(os.path.abspath("../nets/torch"))
sys.path.append(os.path.abspath("../nets/torch/cnns"))
sys.path.append(os.path.abspath("../nets/torch/dnns"))

class Paths:

    @property
    def SIMULATION_DATA(self):
        return "../composite_systems/runs/9_pore/high_strain_rate"

    @property
    def CNNS(self):
        return "../nets/torch/cnns"

    @property
    def DNNS(self):
        return "../nets/torch/dnns"

    @property
    def COMPOSITE(self):
        return "../composite_systems"

    @property
    def CRYSTALLIZATION_DATA(self):
        return "../crystallization/runs"

    @property
    def MODEL_GENS(self):
        return "../nets/model_gens"

    @property
    def NETS(self):
        return "../nets/"

    @property
    def MODEL_DATA(self):
        return "../nets/data"



if __name__ == "__main__":
    pass
