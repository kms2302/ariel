"""
CO-EVOLUTION of a robot's body and its brain

1.  They are INTERDEPENDENT: A good controller for one body might be 
    terrible for another, and a particular body shape might only perform 
    well with a specific control strategy.

2.  Our GENOTYPE consists of TWO PARTS:
    A.  BODY GENOME
    B.  BRAIN GENOME

3.  A random BODY is a list of 3 vectors (each has size GENOTYPE_SIZE=64)

4.  The MAIN DIFFICULTY: the variable I/O size of the controller: different 
    random bodies of the same GENOTYPE_SIZE require different input_size 
    and output_size.

5.  The SOLUTION, ensure that:
    A.  the BRAIN GENOME only specifies the weights and structure of 
        the network.
    B.  the dimensions (input/output size) are determined by the body and 
        the simulation environment before the neural network is constructed.

6.  The 2-STAGE DECODING process:
    A.  Decode BODY GENOME
    B.  Decode BRAIN GENOME using DIRECT SLICE MAPPING.

7.  CMA-ES requires a FIXED GENOME SIZE N:
        N = LEN_BODY_GENOME + LEN_BRAIN_GENOME

8.  DECODING the BODY GENOME from genome x:
    A.  EXTRACT the body genome: x[0:LEN_BODY_GENOME)]
    B.  DECODE the body genome into core and graph
    C.  DEFINE the body in Mujoco (model, data)
    D.  QUERY the model to retrieve the I/O size
            - input_size: len(data.qpos)
            - output_size: model.nu
    E.  CALCULATE LEN_REQUIRED: the total number of weights needed for the 
        network based on input_size and output_size.
            - L1 = input_size * hidden_size
            - L2 = hidden_size * hidden_size
            - L3 = hidden_size * output_size
            - total LEN_REQUIRED = L1 + L2 + L3

9.  DECODING the BRAIN GENOME using DIRECT SLICE MAPPING. Since the required 
    number of weights varies, you can't simply take a fixed-length string 
    of genes and assign them directly:
    A.  END_IDX = LEN_BODY GENOME + LEN_REQUIRED
    B.  EXTRACT the CONTROLLER WEIGHTS: x[LEN_BODY_GENOME:END_IDX]
    C.  UNUSED GENES: Since the total genome length, N, must be fixed, 
        the genome x passed by CMA-ES will be longer than END_IDX. 
        The genes beyond END_IDX are unused genes for that individual.
    D.  CONSTRUCT the CONTROLLER: Map the extracted controller weights W into
        the Input*Hidden and Hidden*Output matrices.
            - Map W1:
                - Initialize idx pointer: idx = 0
                - Flat W1 = W[idx:idx+L1]
                - W1 = Flat W1.reshape((input_size, hidden_size))
                - Update idx pointer: idx=idx+L1
            - Map W2:
                - Flat W2 = W[idx:idx+L2]
                - W2 = Flat W2.reshape((hidden_size, hidden_size))
                - Update idx pointer: idx=idx+L2
            - Map W3:
                - Flat W3 = W[idx:idx+L3]
                - W3 = Flat W3.reshape((hidden_size, output_size))

10. DETERMINING the total genome size N: balancing robustness (making sure 
    every possible body can have a brain) and efficiency (not wasting too 
    many genes).
    A.  N = LEN_BODY_GENOME + LEN_BRAIN_GENOME = 192 + 584 = 776
    B.  LEN_BODY_GENOME = 3 * GENOTYPE_SIZE = 3 * 64 = 192
    C.  LEN_BRAIN_GENOME = (I_MAX*8)+(8*8)+(8*O_MAX) = 584
        - I_MAX (maximum input_size): 35
        - O_MAX (maximum output_size): 30
"""
