<!-- Page 1 -->

1

# Multi-Injection and Multi-Target Probing-Based Load Modeling via a

# Sequential Rolling Identification Framework

Haonan Fang, Graduate Student Member, IEEE, Xinran Zhang, Senior Member, IEEE, Ying Wang, Member, IEEE, Chao Lu, Senior Member, IEEE

Abstract—Accurate load modeling is essential in power system dynamic simulation, stability analysis, and operational security assessment. With the increasing penetration of controllable re sources in modern power systems, probing-based load modeling provides a novel approach for effectively exciting load dynamics. Against this background, this paper proposes a probing-based load modeling method for multi-injection and multi-target load modeling scenarios. The proposed sequential rolling identifica tion framework organizes multi-bus modeling tasks into target specific signal design and identification processes, leading to hierarchical probing-intensity allocation among injection buses. A system-impact-aware multi-injection probing signal optimization model is then established to achieve an adjustable trade-off between identification performance and probing-induced voltage disturbance. Moreover, a consistent probing-band data extraction strategy is developed to reduce the mismatch between the idealized signal design assumptions and practical identifica tion conditions distorted by ambient disturbances. Case studies demonstrate that the proposed method enhances the consistency between theoretical prediction and practical identification results, improves load identification performance, and enables the flexible coordination between identification accuracy and system impact.

Index Terms—Load modeling, parameter identification, active probing, multi-injection and multi-target modeling, power system dynamics.

I. INTRODUCTION L OAD models in power system simulations represent the aggregated behavior of diverse end-use components and constitute an integral part of power system models. There fore, accurate load models are essential for reliable dynamic simulations, which further support the subsequent stability analysis and operational security assessment \[1\]–\[3\]. With the increasing load scale and the rapid development of emerging load components, load dynamics have become more diverse and time-varying, further highlighting the requirement for accurate load modeling in power system studies. Measurement-based load modeling is an important approach for establishing load models. Its basic idea is to identify the parameters of a predefined load model structure from dynamic data, so that the model output can match the observed response as closely as possible \[4\], \[5\]. From an implemen tation perspective, its performance depends mainly on three key elements: the model structure, the identification method, and the data used for identification. First, the predefined model structure determines the scope of load dynamics that can be represented. To accommodate increasingly diverse load compositions and dynamic behaviors, researchers have

This work was supported in part by extended the classical ZIP+IM load model \[6\], \[7\] toward more detailed composite load models, such as the composite load model with distributed generation (CMPLDWG) \[8\], \[9\], while time-varying \[10\] and neural-network-based models \[11\] have also been explored. Second, the identification method determines how the model parameters can be reliably and efficiently estimated. Since load parameter identification is often complicated because of practical difficulties such as increasing parameter dimensionality, parameter coupling, and local optima, methods such as gradient-based optimization \[12\], heuristic optimization \[13\], deep learning \[14\], \[15\], and Bayesian estimation \[16\] have been adopted to improve identification accuracy, computational efficiency, and robust ness. Finally, the data used for identification provide the source from which load dynamic information can be well extracted. However, insufficient response excitation may limit the attainable identification performance. Therefore, the relia bility of measurement-based load modeling is also determined by the availability and quality of disturbance data and the informativeness of the resulting responses. In most load modeling studies and practices, measurement based load modeling is typically performed using large dis turbances \[3\]–\[7\]. These disturbances mainly refer to severe events such as faults and load shedding, which induce strong system responses. Their strong excitation usually provides favorable conditions for identification. However, large distur bances are highly event-dependent. Their infrequent and uncer tain occurrence makes it difficult to satisfy the need for track ing time-varying load properties. In addition, although stronger excitation is beneficial for parameter identification, it may also impose non-negligible stress on system operation \[17\]. To reduce the dependence on large-disturbance data, ambient signal-based load modeling methods have been further devel oped \[19\], \[20\]. Ambient signals are defined as continuous low-level fluctuations caused by load variations, renewable power fluctuations, and other daily normal operating factors. Compared with large disturbances, ambient signals have better continuity and data availability, thereby enabling periodic load modeling \[18\]. However, their excitation may be weaker or more concentrated in limited frequency ranges, which may affect the reliability of the identification results. Overall, both large disturbances and ambient signals are uncontrollable, which motivates the exploration of a more adjustable approach for generating the data used for identification. The increasing penetration of controllable resources, such as distributed energy resources, energy storage systems, flexible loads, and FACTS devices, provides more available channels for power systems, thereby laying a physical basis for actively


<!-- Page 2 -->

2

generating low-level disturbances \[21\], \[22\]. Against this back ground, probing-based identification has emerged as a promis ing load modeling approach, which injects designed probing signals through controllable resources to excite more desirable load dynamics and then utilizes the resulting data for model identification \[23\]. Unlike the passive use of large disturbances and ambient signals, active probing exploits controllable re sources and thus enables flexible design of disturbance timing, frequency components, and perturbation intensity. Currently, active probing has been applied in various power-system iden tification and modeling tasks, including mode identification \[24\], \[25\], inertia estimation \[26\], impedance measurement \[27\], and converter dynamic modeling \[28\], demonstrating its feasibility. Moreover, the use of different probing waveforms in these studies, such as multisine \[29\], pseudo-random binary sequence \[30\], and chirp signals \[31\], further reflects the design flexibility of probing signals. Our previous work made an initial effort to develop a probing-based load modeling approach \[32\]. It established an identification-error model to guide the optimization of low level probing signals to improve load identification perfor mance. However, as an initial exploration, this work was confined to a single-injection and single-target-load setting, leaving the practical scenario with multiple controllable re sources and multiple load-modeling targets insufficiently con sidered. In addition, the system impact of probing, which is another important concern, was not incorporated into the design objective. Therefore, extending probing-based load modeling to multi-injection and multi-target scenarios with system-impact awareness can improve its practical applicabil ity and performance potential. Nevertheless, this extension also leads to more complex problem characteristics. Multi-injection leads to unequal contributions of different injection buses to identification and system impact, making probing-intensity allocation nontrivial. Meanwhile, multiple target loads further imply different modeling requirements, calling for coordinated organization of target-specific probing tasks. Therefore, multi injection probing-based load modeling is not a direct extension of single-injection signal design, but requires coordinated treatment of target-specific task organization, probing-intensity allocation among different injection buses, and the balance between identification performance and system impact. Motivated by these challenges, a multi-injection probing based load modeling method is developed in this paper, with the main contributions summarized as follows: 1) A sequential rolling identification framework is proposed for multi-injection probing-based load modeling. This framework decomposes the modeling tasks of multiple load buses into a series of target-specific signal design and identification processes, enabling hierarchical probing intensity allocation according to target-dependent require ments and the different effects of injection locations. 2) A system-impact-aware multi-injection probing signal optimization model is established. The model is formu lated by incorporating the probing-to-load-voltage trans fer characteristics and the frequency-domain identifica tion error model. Meanwhile, the optimization objective explicitly includes both the theoretical identification error and the probing-induced system-side voltage disturbance, enabling an adjustable trade-off between identification performance and system impact. 3) To maintain consistency between theoretical signal design and practical identification under ambient disturbances, a probing-band data extraction strategy is developed. The influence of out-of-band components on parameter identification is first analyzed, revealing their adverse effects on the identification conditions assumed in signal optimization. On this basis, consistent filtering is applied to restore the intended identification conditions, thereby preserving the validity of the theoretical signal design. The remainder of this paper is organized as follows. Section II presents the sequential rolling identification framework for multi-injection probing-based load modeling. Section III establishes the multi-injection probing signal optimization model. Section IV introduces the consistent probing-band data extraction strategy. Section V provides case studies, and Section VI concludes this paper.

II. SEQUENTIAL ROLLING IDENTIFICATION FRAMEWORK FOR PROBING-BASED LOAD MODELING Probing-based load modeling is an approach that actively excites load dynamics through designed signal injections and then identifies load models from the resulting responses. To extend this approach to multi-device and multi-target load scenarios, a sequential rolling identification framework is proposed in this paper, as shown in Fig. 1. The basic idea of this framework is to sequentially handle multiple load buses to be identified, where probing signals from multiple injection buses are jointly designed for one target load bus at each stage. Since different injection buses contribute differently to the excitation of load responses, targeting only one load bus at a time in this framework avoids the necessity for all injection buses to generate strong excitation, thereby reducing the impact on the power system. Overall, the process of probing-based load modeling with multiple signal injections can be described as follows: 1) Target-specific Probing Signal Design: Specifically, in this procedure, a series of independent optimization problems are formulated for successive stages, with each problem jointly determining the probing signals at all available injection buses for the current target load bus and yielding a coordinated multi-injection signal set. This target-oriented signal design plays a critical role in the proposed framework because the probing signals determine the excitation characteristics of the target load bus. In this paper, the multisine signal is chosen as the base waveform, as it permits the flexible combination of frequency components to target specific dynamics of interest, which is defined as:

u(t) = ∑︂m

i=1 A(ωi) sin(ωit + φ(ωi)) (1)

where m is the number of frequency components, A(ωi) and φ(ωi) represent the amplitude and phase of the i th frequency component ωi , respectively. 2) Multi-injection Probing: The designed probing signals, tailored for different load


<!-- Page 3 -->

3

Fig. 1. Sequential rolling identification framework

buses, are sequentially dispatched to the corresponding con trollable devices, which adjust their outputs simultaneously to excite the current target load. The resulting data are then collected for subsequent processing and identification. 3) Consistent Filtering: The collected voltage and power data are processed using the same filter to retain the probing-band components and suppress out-of-band disturbances. The filtered data are then used for load model identification. 4) Load Model Identification: In this step, the parameters of the current target load model are estimated from the collected data. Specifically, load model identification refers to finding adequate load model parameters to make the model output match the measurements as closely as possible, which can be expressed as: θ = arg min θ ∑︂ (Ym(t) − Yp(t|θ))2 (2)

where t indexes time over the identification interval, θ denotes the load model parameters, Ym(t) is the measured power output, and Yp(t|θ) is the model-predicted power output. Given the richness and maturity of existing research on identification methods, a hierarchical identification framework is adopted in this work to improve identification efficiency, with further details provided in \[20\].

III. OPTIMIZATION MODEL OF PROBING SIGNALS

Probing signal design may be cast as an optimization problem for improving identification performance. Its formula tion requires characterization of the signal excitation process. Specifically, as shown in Fig. 2, this process consists of two cascaded channels: the probing-signal-to-load-voltage channel governed by the transmission network and the load-voltage to-load-power channel determined by the load model charac teristics. These two channels are analyzed below to establish the multi-injection probing signal optimization model.

A. Sensitivity-Based Modeling of the Probing-Signal-to-Load Voltage Channel The probing-signal-to-load-voltage channel describes the mapping from power injections at the injection buses to voltage variations at the load buses. Since this mapping is Fig. 2. Probing excitation process primarily governed by the transmission network, it is approx imated using static sensitivity coefficients derived from the power flow equations: SP rj = ∂Vr ∂Pj , SQrj = ∂Vr ∂Qj (3)

where Vr is the voltage magnitude at bus r, and Pj , Qj are the power injections at bus j. In practice, probing signals are expected to remain small to avoid noticeable effects on system operation. Under such small disturbances, the system can be approximately treated as linear, so different frequency components remain decoupled and the effects of probing signals from different injection buses are additive. Thus, the voltage perturbation ∆Vr(ωi) at frequency ωi can be expressed as: ∆Vr(ωi) = ∑︂

j (SP rj∆Pj (ωi) + SQrj∆Qj (ωi)) (4)

where ∆Pj (ωi) and ∆Qj (ωi) denote the active and reactive power disturbances injected at bus j, respectively. Although static power flow and dynamic simulation do not impose identical constraints, they share the same network topology and parameters. Therefore, the static sensitivities can still provide a reasonable approximation of the channel gain, as later supported by the voltage spectra.

B. Identification Error Modeling of the Load-Voltage-to-Load Power Channel 1) Load Model Structure: In this paper, the ZIP+IM model is adopted as the basis for the subsequent analysis. As a simple and widely recognized composite load model, it is an appropriate choice for a preliminary study. The ZIP+IM model consists of a constant torque third-order induction motor (IM) and a ZIP load. To

<!-- Image omitted -->

<!-- Image omitted -->


<!-- Page 4 -->

4

facilitate parameter identification, the conventional third-order IM model is reformulated into the transformed and simplified form presented in \[20\], as given by: { dFd dt = −bFd + sω0Fq + aVd dFq dt = −bFq − sω0Fd + aVq ds dt = 1 H2 (Tm − VqFd + VdFq) (5)

where a = 1 Td0X′ ( X X′ − 1), b = X Td0X′ , Fd = Ed/X′ , Fq = Eq/X′ . X, X′ denote the rotor open-circuit reactance and transient reactance, respectively. Td0 is the rotor open-circuit time constant, H2 is the inertia time constant, and Tm is the constant mechanical torque. In addition, Vd, Vq are the d-axis and q-axis components of the voltage phasor, re spectively. Ed, Eq are the corresponding components of the electromotive-force phasor. s is the slip ratio. The total load power is obtained by summing the ZIP load and IM components: {︄ P = FdVq − FqVd + PzV 2 + PiV + Pp

Q = −VdFd − VqFq + QzV 2 + QiV + Qp (6)

where \[Pz, Pi , Pp, Qz, Qi , Qp\] are the ZIP load parameters. 2) Identification Error Model: In our previous work \[32\], a frequency-domain identification error model was established to characterize the effects of input voltage perturbations and noise on parameter identification. For m probing frequencies and n model parameters, this model gives the variance of the parameter estimation error as: Var(∆θ) = Var(HN) = (H ◦ H) Var(N) (7) where Var(·) denotes the elementwise variance, ∆θ ∈ R n×1 is the parameter estimation error, N ∈ R 4m×1 is the frequency domain noise vector associated with the active- and reactive power responses, and ◦ denotes the Hadamard product. The matrix H ∈ R n×4m is determined by the parameter sensitivity matrix J as H = (J T J) −1J T . Specifically, J ∈ R 4m×n characterizes the sensitivities of the active- and reactive-power responses to the model parameters at the probing frequencies, and its elements are defined as follows: J4i−3,j = ∆V (ωi) ∂Kp(ωi , θr) ∂θj (8)

J4i−2,j = ∆V (ωi) ∂φpG(ωi , θr) ∂θj Kp(ωi , θr) (9)

J4i−1,j = ∆V (ωi) ∂Kq(ωi , θr) ∂θj (10)

J4i,j = ∆V (ωi) ∂φqG(ωi , θr) ∂θj Kq(ωi , θr) (11)

where 1 ≤ i ≤ m and 1 ≤ j ≤ n. Moreover, ∆V (ωi) is the voltage-perturbation amplitude at frequency ωi , and θr is the reference parameter vector. Kp and φpG denote the magnitude and phase angle of the active-power transfer function, respectively, while Kq and φqG are those of the reactive-power transfer function. These transfer functions are obtained based on the preceding load model in (5), with the detailed derivation provided in our previous work \[32\]. C. Multi-Injection Probing Signal Optimization

By combining the network-side voltage transfer character istics with the load-side identification error model, a target oriented multi-injection optimization model is established. For each target load bus, the probing signals at all available injection buses are jointly designed to balance identification performance and system impact. For the current target load bus k, identification performance is evaluated by the weighted parameter-error variance: Objid,k = WkVar(∆θk) = Wk(Hk ◦ Hk)Var(Nk) (12) where Wk ∈ R 1×n is the weighting vector, and ∆θk, Hk, and Nk denote the corresponding quantities for load bus k. Meanwhile, the system impact is characterized by the voltage-perturbation over the monitored buses:

Objsys = ∑︂p

r=1 ∑︂m

i=1 |∆Vr(ωi)| 2 (13)

where p is the number of considered buses. Owing to the time–frequency energy equivalence established by Parseval’s theorem, this metric also reflects the time-domain disturbance intensity while allowing direct evaluation using the frequency domain channel model. Moreover, the quadratic form im poses a greater penalty when voltage perturbations are more concentrated, enabling this metric to discourage excessive disturbances and undue localization. The two objectives are combined to balance target-load identification and probing-induced system impact: Obj = Objid,k + βObjsys (14) where β is a weighting factor. To limit the probing magnitude at each injection bus, the following frequency-domain constraint is imposed:

∑︂m

i=1 ∆Qj (ωi) ≤ Athre,j , ∆Qj (ωi) ≥ 0 (15)

where Athre,j is the probing-amplitude limit at injection bus j, and ∆Qj (ωi) is the perturbation magnitude at ωi . Since the time-domain magnitude of a multisine signal is bounded by the sum of its component amplitudes, this con straint provides a conservative and readily tractable limit with out explicitly evaluating the waveform peak. Only reactive power injections are considered as they generally provide more effective voltage excitation. The complete target-oriented multi-injection optimization model is formulated as follows:

min {ωi,∆Qj (ωi)} Wk(Hk ◦ Hk) Var(Nk) + β ∑︂p

r=1 ∑︂m

i=1 |∆Vr(ωi)| 2

s.t. { ∆Vr(ωi) = ∑︂q

j=1 SQrj∆Qj (ωi),

∑︂m

i=1 ∆Qj (ωi) ≤ Athre,j , ∆Qj (ωi) ≥ 0,

ωmin ≤ ωi−1 < ωi ≤ ωmax, 2 ≤ i ≤ m. (16) where 1 ≤ i ≤ m, 1 ≤ j ≤ q, and 1 ≤ r ≤ p, with q denoting the number of injection buses. Phase coordination is not included in this study, and thus the phases of the multisine components are fixed and aligned across injections.


<!-- Page 5 -->

5

To solve the resulting optimization model, the two-stage strategy in \[32\] is adopted. First, the probing frequencies are optimized with the reactive-power component amplitudes fixed, after which the amplitudes are optimized at the selected frequencies. Moreover, to improve computational efficiency, the frequencies are selected from a predefined discrete set, which allows the frequency-dependent sensitivity terms in J to be precomputed and reused during optimization. The differential evolution algorithm \[33\] is used throughout the two-stage optimization owing to its global search capability and convenient implementation.

IV. IDENTIFICATION DATA PREPARATION FOR PROBING-BASED LOAD MODELING The signal design model in this paper is established based solely on the probing-induced frequency components. How ever, in practice, ambient disturbances can also introduce additional components into both the voltage input and power responses, causing the actual identification data to deviate from the conditions considered in signal design. This section discusses these influences and presents a consistent probing band data extraction strategy, whose role is not limited to disturbance suppression but is primarily to restore consistency between signal design and practical identification.

A. Effects of Out-of-Band Disturbances on Identification As illustrated in Fig. 2, the voltage input used for identifi cation contains both the probing-induced component and the component introduced by ambient disturbances. Accordingly, the corresponding power response can be expressed as: Y˙m(ω) = GL (︁ jω, θr )︁\[︁ V˙ p(ω) +V˙ o(ω) \]︁ + N˙ p(ω) + N˙ o(ω) (17) where the subscripts p and o denote components within the probing band and additional out-of-band components intro duced by ambient disturbances, respectively. Here the addi tional voltage component induced by ambient disturbances in the probing band is neglected because this band is generally dominated by the designed excitation. Based on the identification error model in (7), the poten tial influence of the out-of-band components on parameter identification can be examined. Since the model is assembled from frequency-domain components, it can be expanded to explicitly show the probing-band and out-of-band terms: ∆θaug = (︁ J T 0,pΣpJ0,p + J T 0,oΣoJ0,o )︁−1

× (︁ J T 0,pVT p Np + J T 0,oVT o No )︁ , (18)

where Vp and Vo are diagonal matrices composed of the voltage magnitudes at the probing-band and out-of-band fre quencies, respectively; J0,p and J0,o denote the corresponding model-dependent parts of the sensitivity matrix; and Np and No are the associated frequency-domain noise vectors. Moreover, Σp = VT p Vp and Σo = VT o Vo. The out-of-band voltage component Vo affects both J T 0,oΣoJ0,o and J T 0,oVT o No, so its overall influence on the parameter error can be non-monotonic, with the error first increasing and then decreasing as the strength of Vo increases. In practical cases, ambient disturbances typically produce pronounced out-of-band power fluctuations accompanied by relatively weak voltage components, a combination that tends to aggravate identification errors. This observation motivates the removal of out-of-band components before identification.

B. Consistent Probing-Band Data Extraction To mitigate the adverse influence of out-of-band compo nents, a filtering-based method is adopted before parameter identification, as illustrated in Fig. 3. This method extracts the voltage inputs and power responses within the designed probing band while preserving their correspondence.

Fig. 3. Block diagram of filtering-based method Specifically, the same filter is applied to the voltage inputs and power responses. Let the frequency response of the filter be denoted by F(jω). The filtered quantities are given by: V˙ f (ω) = F(jω)V˙ (ω), Y˙ f (ω) = F(jω)Y˙ (ω). (19) Considering the frequency-domain load model, Y˙ (ω) = GL(jω, θ)V˙ (ω) + N˙ (ω), (20) where Y˙ (ω) denotes the active or reactive power response. The filtered output can then be written as Y˙ f (ω) = F(jω)GL(jω, θ)V˙ (ω) + F(jω)N˙ (ω) = GL(jω, θ)V˙ f (ω) + F(jω)N˙ (ω). (21)

This indicates that applying the same filter to the voltage inputs and power responses preserves the input–output rela tionship of the load model, while attenuating the components outside the probing band. Overall, the proposed data processing strategy should be interpreted as a data consistency-restoring procedure rather than a conventional noise reduction operation. By extracting the probing-band components while preserving the corre spondence between voltage inputs and power responses, the mismatch between practical identification conditions and the assumptions adopted in the probing-signal design model is reduced. Otherwise, violated assumptions would undermine the effectiveness of the proposed probing-based approach.

V. CASE STUDIES A. Simulation Introduction To validate the proposed probing signal design method for load modeling under multi-injection scenarios, the IEEE 39- bus system augmented by two attached subnetworks is adopted in this paper as the simulation system, as shown in Fig. 4. Considering both symmetry and diversity, Bus 41, 42, 56, and 57 are selected as the target load buses whose models are to be identified. These target buses are modeled as ZIP+IM loads, with their parameters and initial conditions listed in Table I. Controllable devices are assumed to be deployed at Bus 47, 51,

<!-- Image omitted -->


<!-- Page 6 -->

6

Fig. 4. Structure of test system TABLE I LOAD MODEL PARAMETERS AND INITIAL CONDITIONS OF THE BUSES Bus a b H Tm ZIP ratio P0 Q0 41 185.2 33.52 5.00 0.800 2: 6: 2 / 7: 2: 1 1.0 0.48 42 115.2 33.44 1.40 0.300 2: 6: 2 / 7: 2: 1 0.5 0.18 56 115.2 33.44 1.40 0.600 2: 6: 2 / 7: 2: 1 1.0 0.18 57 185.2 33.52 5.00 0.400 2: 6: 2 / 7: 2: 1 0.5 0.08

62, and 66, each allowed to inject reactive-power perturbations with a total intensity no greater than 0.1 p.u. Ambient disturbances are also included to emulate realistic operating conditions. Following our previous work \[32\], in which practical ambient fluctuations were shown to be mainly concentrated in the low-frequency range, they are modeled as a mixture of low-pass-filtered white noise with cutoff frequencies of 0.2, 1.5 and 15 Hz, weighted by 60%, 20%, and 20%, respectively. These disturbances are introduced at all buses in the two subnetworks except the probing-signal injection buses, with a standard deviation of 3 × 10−4 p.u. Simulations are carried out in MATLAB/PSAT on a computer with an AMD Ryzen 7 4800H CPU (2.9 GHz) and 32 GB RAM. Each case lasts 20 s with the time step being 0.01 s. Based on the above settings, the signal design model is con structed. For the probing-signal-to-load-voltage channel, the sensitivity coefficients are obtained by perturbing the reactive power injection in the power flow equations and calculating the resulting voltage variations. The matrix H is obtained based on the reference parameters of each target load bus according to its definition. The noise variance Var(N) is determined from the spectral variance profile of the simulated ambient disturbances. To analyze the trade-off between identification performance and system impact, three weighting conditions with β ⋆ = 0, 100, and 400 are considered, where β ⋆ is the scaled representation of β with respect to the baseline objective of each target load bus. The upper frequency limit of this signal design model is set to 10 Hz, with the number of frequency components fixed at m = 10.

B. Multi-Target Probing Signal Design Results Based on four target load buses and three β ⋆ settings, a total of 12 independent signal design cases are obtained. The corresponding objective values are summarized in Table II, where “Reference” refers to the baseline design with uniformly distributed frequencies and equal reactive-power amplitudes. TABLE II SUMMARY OF SIGNAL DESIGN RESULTS Target bus Reference β ⋆ = 0 β ⋆ = 100 β ⋆ = 400

41 Id: 7.05e4 V: 3.60 Id: 343.5 V: 4.380 Sum: 343.5 Id: 416.3 V: 3.188 Sum: 735.1 Id: 562.3 V: 2.082 Sum: 1395.1

42 Id: 6.85e5 V: 3.60 Id: 3667 V: 4.026 Sum: 3667 Id: 4312 V: 3.206 Sum: 7503 Id: 7206 V: 1.777 Sum: 14283

56 Id: 1.61e5 V: 3.60 Id: 1292 V: 4.549 Sum: 1292 Id: 1457 V: 3.412 Sum: 2626 Id: 2334 V: 2.092 Sum: 5203

57 Id: 3.63e5 V: 3.60 Id: 1967 V: 4.507 Sum: 1967 Id: 2308 V: 3.348 Sum: 4134 Id: 2961 V: 2.566 Sum: 8560 Id, V, and Sum denote Objid, Objsys, and Obj, respectively.

Compared with the reference design, the optimized results achieve substantially lower objective values for all target load buses, demonstrating the effectiveness of the proposed method. For illustration, Fig. 5 and Fig. 6 show the frequency-domain allocation and time-domain waveforms of a representative signal design case. The designed probing signals for differ ent target load buses are then sequentially injected into the system to excite load responses, based on which load model identification is carried out.

Fig. 5. Frequency-domain allocation of the multi-injection probing signals designed for Bus 41

Fig. 6. Time-domain waveforms of the multi-injection probing signals designed for Bus 41

C. Alignment Between Identification Conditions and Signal Design Assumptions In simulations, ambient disturbances may introduce addi tional components into the identification data, causing devia tions from the assumptions adopted in the signal design stage. Fig. 7 presents the simulated, expected, and filtered voltage spectra at target load bus 41 after probing injection. At the designed probing frequencies, the simulated spectrum agrees well with the expected spectrum, confirming the validity of the network channel model. Meanwhile, additional low-frequency

<!-- Image omitted -->

<!-- Image omitted -->

<!-- Image omitted -->


<!-- Page 7 -->

7

TABLE III COMPARISON OF THEORETICAL AND IDENTIFIED VARIANCES FOR THE REPRESENTATIVE CASE (TARGET BUS 41, β ⋆ = 100) Param. Bus 41 Bus 42 Theo. Raw Filt. Theo. Raw Filt. a 277.857 196.021 234.451 418.345 137.722 347.942 b 27.938 7.406 24.191 166.673 44.537 168.504 H 9.509 4.672 9.111 15.521 4.313 15.931 Tm 1.000 1.000 1.000 0.594 0.328 0.423 Param. Bus 56 Bus 57 Theo. Raw Filt. Theo. Raw Filt. a 283.494 92.984 195.210 839.527 470.438 612.855 b 106.909 32.220 94.645 75.869 17.380 48.127 H 9.757 2.947 9.720 28.832 11.096 20.876 Tm 0.884 0.467 0.886 1.552 1.232 1.255 Theo. denotes the theoretical variances, whereas Raw and Filt. denote the identified variances obtained from the unprocessed and filtered data, respectively. components induced by ambient disturbances are observed in the simulated spectrum. Although these components have relatively small magnitudes, their presence results in notice able deviations between the identified variances and theoretical predictions, as shown in Table III.

Fig. 7. Comparison of voltage spectra at target load bus 41 To reduce the influence of these additional components, the consistent probing-band data extraction strategy is applied. As shown in Table III, the identified variances obtained from the extracted data become closer to the theoretical values than those from the unprocessed data, suggesting that the proposed strategy improves the agreement between practical identification results and theoretical predictions. Furthermore, Fig. 8 shows that the identification errors are reduced after data extraction, indicating an additional benefit for probing-based load identification.

Fig. 8. Comparison of identification results before and after filtering

D. Effectiveness of the Proposed Signal Design Method 1) Comparison among Different Probing Signals: Based on the signal design results, β ⋆ = 100 is selected as a representative weighting condition to compare the target oriented probing signals with the uniformly distributed probing signal and a simultaneous multi-target design. As a natural alternative, the simultaneous formulation optimizes a common set of probing signals for all target load buses:

min {ωi,∆Qj (ωi)} ∑︂ NT

k=1 wkObjid,k + βObjsys, (22)

where NT is the number of target load buses and wk is inversely proportional to the reference Objid,k under the uniform design. The constraints remain the same as those in (16). All probing signal designs follow the preceding simulation settings, and only the probing signals differ. For each design, 200 independent simulations are performed with independently generated ambient disturbances to obtain statis tical identification results. The identification performance is evaluated using the nor malized parameter error and the composite error index:

∆θi = | ˆθi − θi | |θi | , Ierr = WVar(∆θ), (23)

where ˆθi and θi denote the identified and reference values of the ith parameter, respectively, and the coefficients in W are determined according to the scales of different parameters.

Fig. 9. Normalized parameter errors under different probing signals

Fig. 10. Composite error indices under different probing signals The resulting identification performances are shown in Fig. 9 and Fig. 10. The optimized probing signals consis tently achieve significantly lower identification errors than the uniformly distributed signal, demonstrating the necessity of probing signal optimization. Among the optimized designs, the signal specifically designed for each target load bus generally provides the lowest composite error index at its intended target, while the simultaneous multi-target design maintains favorable identification performance across different target buses. These results indicate that the target-oriented design can tailor the probing excitation more closely to the current load-modeling task, whereas the simultaneous design is constrained by the need to balance the identification requirements of multiple target buses. Since accurate identification of each target load is

<!-- Image omitted -->

<!-- Image omitted -->

<!-- Image omitted -->

<!-- Image omitted -->


<!-- Page 8 -->

8

TABLE IV SYSTEM IMPACT INDICES UNDER DIFFERENT INJECTION SCHEMES Case Max Vpp (p.u.) Mean Vpp (p.u.) VSS (p.u.) Multi-injection 0.0140 0.0066 0.0276 Bus 47 only 0.0364 0.0068 0.0530 Bus 62 only 0.0389 0.0071 0.0665

central to multi-target load modeling, the sequential framework is well suited to such identification tasks. 2) Comparison Between Multi-Injection and Single Injection Probing Signals: To investigate the benefit of spatially distributing probing signals, the multi-injection scheme is compared with two aggregated single-injection schemes. In the multi-injection case, the optimized probing signals obtained under β ⋆ = 100 are injected at Buses 47, 51, 62, and 66. These four signals are then aggregated into a single signal and injected only through Bus 47 or Bus 62 to construct the two single-injection schemes, so that all schemes have comparable total probing intensities. Each scheme is also evaluated based on 200 independent simulations.

Fig. 11. Identification performance comparison between multi-injection and single-injection cases The identification results are shown in Fig. 11, while the cor responding system-impact indices are summarized in Table IV and Fig. 12. Specifically, the single-injection schemes provide uneven excitation across target buses: although concentrating the probing signal at one bus can effectively excite certain targets, it may leave others insufficiently excited and thereby result in substantially larger identification errors, as exempli fied by the Bus-62-only scheme, which yields a particularly large identification error at Bus 42. In contrast, for all target buses, the identification performance of the multi-injection scheme lies between those of the two single-injection schemes and is generally closer to the better-performing one, indicating that spatially distributing the probing signals incurs only a limited loss in identification accuracy. These findings suggest that the multi-injection scheme can provide more balanced excitation across target buses and better accommodate the sequential identification of multiple target buses. More importantly, Table IV shows that the multi-injection scheme has the lowest overall system impact among the compared schemes. Furthermore, to characterize the overall distribution of voltage disturbances, the Vpp values across all monitored buses are ranked in descending order separately for each scheme, as shown in Fig. 12. The ranked distributions reveal that the multi-injection scheme exhibits a substantially lower maximum Vpp and a more even distribution. This Fig. 12. Ranked distributions of bus Vpp

behavior can be attributed to the dispersion of probing intensity among several controllable devices rather than its concen tration at a single bus. Overall, the multi-injection scheme is system-friendly, reducing concentrated voltage disturbances while maintaining comparable identification performance.

Fig. 13. Identification errors under different β ⋆ values

Fig. 14. System-impact indices under different β ⋆ values 3) Probing-Intensity Allocation under Different Weightings in the Sequential Framework: Further insight can be obtained by examining the probing-intensity allocation among the in jection buses under different β ⋆ values, as shown in Fig. 15, where the simultaneous multi-target design is also included for comparison. The total perturbation assigned to injection bus j is denoted by Injj and defined as

Injj = ∑︂m

i=1 |∆Qj (ωi)| (24)

When β ⋆ = 0, Injj remains relatively high at all injection buses to maximize the excitation of the target bus. As β ⋆ increases, however, the attenuation of Injj differs markedly among the injection buses. For example, Inj47 for the nearby target bus 41 and Inj62 for the nearby target bus 56 are largely preserved, suggesting greater contributions from these injec tion buses to the corresponding identification tasks, whereas those assigned to less effective injection buses, such as Inj62 for target bus 41 and Inj47 for target bus 56, undergo rapid attenuation. In other words, the inclusion of the system-impact term does not simply reduce the overall probing intensity uni formly; instead, it promotes a more targeted and differentiated allocation of probing intensity among the injection buses.

<!-- Image omitted -->

<!-- Image omitted -->

<!-- Image omitted -->

<!-- Image omitted -->


<!-- Page 9 -->

9

Fig. 15. Variation of injection-bus probing intensity under different β ⋆ values

This differentiated probing-intensity allocation highlights the importance of combining the multi-injection probing scheme with the sequential identification framework. Since each stage focuses on only one target bus, the probing intensity can be preferentially retained at injection buses with greater contributions to the current identification task and reduced at less effective locations, thereby forming a target-dependent hierarchical allocation. By contrast, the simultaneous design has to accommodate all target buses at once. As shown in Fig. 15, it therefore retains higher probing intensities at multiple injection buses, especially under the larger weighting condition, which reduces the flexibility of probing-intensity allocation. Overall, the sequential framework better exploits the heterogeneous contributions of different injection buses, enabling effective target-specific excitation while avoiding unnecessary probing effort.

E. Impact–performance Comparison with Conventional Measurement-based Methods Finally, the proposed multi-injection active probing method is compared with conventional load modeling based on large disturbances and ambient signals. Specifically, the background ambient-disturbance setting introduced in Section V-A is re tained. For large-disturbance-based load modeling, a three phase short-circuit fault is applied at Bus 14 at t = 1 s and cleared at t = 1.5 s. For the ambient-signal-based comparison, additional ambient signals following the spectral profile in Section V-A, with a standard deviation of 3 × 10−3 p.u., are introduced at Buses 47, 51, 62, and 66. The remaining simulation and identification settings are kept unchanged. For illustration, Fig. 16 presents the response curves and fitting results from one simulation case under each of the two conventional disturbance conditions.

Fig. 16. Representative responses and model fitting Fig. 17 maps the three methods onto a normalized impact– error plane. For each target bus, Vss and Ierr are normalized by the corresponding active-probing results under β ⋆ = 100 TABLE V IMPACT–ERROR INDEX PAIRS (Vss, Ierr) OF DIFFERENT METHODS Method Bus 41 Bus 42 Bus 56 Ambient (0.0022, 0.9835) (0.0022, 5.0680) (0.0022, 3.3732) β ⋆ = 0 (0.0390, 0.0007) (0.0340, 0.0087) (0.0385, 0.0032) β ⋆ = 100 (0.0276, 0.0010) (0.0266, 0.0106) (0.0297, 0.0035) β ⋆ = 400 (0.0174, 0.0013) (0.0148, 0.0158) (0.0188, 0.0059) Large dist. (0.8899, 0.0005) (0.8899, 0.0418) (0.8899, 0.0110) Method Bus 57 Normalized center (¯rV , r¯I ) Ambient (0.0022, 2.0820) (0.0764, 716.36) β ⋆ = 0 (0.0399, 0.0037) (1.3233, 0.8475) β ⋆ = 100 (0.0305, 0.0036) (1.0000, 1.0000) β ⋆ = 400 (0.0235, 0.0047) (0.6430, 1.4397) Large dist. (0.8899, 0.0030) (31.165, 1.5392) Ambient and Large dist. denote ambient-signal and large-disturbance methods; r¯V and r¯I are the geometric means of normalized Vss and Ierr.

and expressed as base-10 logarithmic ratios. Meanwhile, the original indices and normalized centers are summarized in Table V. It can be observed that the three methods oc cupy clearly different regions in the normalized plane. The ambient-signal-based results are mainly located in the upper left region, indicating low system impact but relatively large identification errors. The large-disturbance-based results are concentrated in the lower-right region, reflecting the fact that strong fault excitation facilitates parameter identification but causes a substantial system impact. The active-probing results are also concentrated in the lower part of the plane but are shifted markedly toward the low-impact side compared with the large-disturbance results, indicating that the proposed method achieves identification performance comparable to that of the large-disturbance-based method while imposing consid erably less system impact. This favorable impact–performance balance can be attributed to the controllability and design flexibility of the probing signals, which allow the excitation to be tailored for identification, while the probing-band data extraction further improves identification performance by re ducing the effects of out-of-band components.

Fig. 17. Impact–error comparison of different identification methods Accordingly, the proposed method provides an explicit and controllable mechanism for coordinating identification perfor mance and system impact through probing-signal optimization, whereas conventional methods have difficulty doing so.

VI. CONCLUSION This paper develops a multi-injection probing-based load modeling method for multi-target scenarios, including a se quential rolling identification framework to organize target specific identification tasks, a system-impact-aware probing

<!-- Image omitted -->

<!-- Image omitted -->

<!-- Image omitted -->


<!-- Page 10 -->

10

### signal optimization model to coordinate identification perfor

### mance and system impact, and a consistent probing-band data

### extraction method to align practical identification conditions

### with the signal design assumptions. Case studies validate the

### method from both the injection and identification perspectives.

### From the injection-side perspective, compared with single

### injection schemes, this multi-injection design avoids excessive

### disturbance concentration at one location and distributes the

### system impact more evenly. On this basis, the proposed

### sequential rolling identification framework further exploits

### the heterogeneous contributions of different injection buses

### and promotes hierarchical probing-intensity allocation, which

### avoids unnecessarily degrading identification performance by

### uniformly weakening all injections. From the identification

### side perspective, the probing-band data extraction restores

### design–identification consistency and preserves the effective

### ness of the theoretical design. Then, the target-oriented opti

### mization model determines probing signals that balance iden

### tification effectiveness and system impact, with each target

### specific design generally achieving the lowest error at its

### intended bus and clearly outperforming the uniform design.

### Overall, by combining coordinated multi-injection probing

### with target-oriented signal design, the proposed method re

### duces the dependence of load modeling on uncontrollable

### disturbances and offers a more flexible and task-adaptive

### modeling approach. These results point toward a broader role

### for probing-based load modeling in load-model construction.

### Future work will extend the framework to more complex com

### posite load models and advance its practical implementation.

REFERENCES

\[1\] A. Arif, Z. Wang, J. Wang, B. Mather, H. Bashualdo, and D. Zhao, “Load modeling–A review,” IEEE Trans. Smart Grid, vol. 9, no. 6, pp. 5986– 5999, Nov. 2018. \[2\] I. D. Pasiopoulou, E. O. Kontis, T. A. Papadopoulos, and G. K. Papagian nis, “Effect of load modeling on power system stability studies,” Electr. Power Syst. Res., vol. 207, Art. no. 107846, Jun. 2022. \[3\] M. Garmroodi, D. J. Hill, G. Verbic, and J. Ma, “Impact of load dynamics on electromechanical oscillations of power systems,” IEEE Trans. Power Syst., vol. 33, no. 6, pp. 6611–6620, Nov. 2018. \[4\] H. Renmu, M. Jin, and D. J. Hill, “Composite load modeling via measurement approach,” IEEE Trans. Power Syst., vol. 21, no. 2, pp. 663–672, May 2006. \[5\] B. Tan, J. Zhao, N. Chiang, and N. Duan, “High-dimension Bayesian parameter estimation for WECC composite load model using realistic event measurements,” IEEE Trans. Power Syst., vol. 40, no. 5, pp. 3979– 3992, 2025. \[6\] J. Xie et al., “Imitation and transfer Q-learning-based parameter identi fication for composite load modeling,” IEEE Trans. Smart Grid, vol. 12, no. 2, pp. 1674–1684, Mar. 2021. \[7\] Y. Lin, Y. Wang, J. Wang, and D. Shi, “Tensor-based parameter reduction of dynamic load models with variable frequency drive,” IEEE Trans. Power Syst., vol. 37, no. 2, pp. 1091–1101, Mar. 2022. \[8\] Z. Ma, Z. Wang, Y. Wang, R. Diao, and D. Shi, “Mathematical represen tation of WECC composite load model,” J. Modern Power Syst. Clean Energy, vol. 8, no. 5, pp. 1015–1023, Sep. 2020. \[9\] F. Bu, Z. Ma, Y. Yuan, and Z. Wang, “WECC composite load model parameter identification using evolutionary deep reinforcement learning,” IEEE Trans. Smart Grid, vol. 11, no. 6, pp. 5407–5417, Nov. 2020. \[10\] M. Cui, J. Wang, Y. Wang, R. Diao, and D. Shi, “Robust time-varying synthesis load modeling in distribution networks considering voltage disturbances,” IEEE Trans. Power Syst., vol. 34, no. 6, pp. 4438–4450, Nov. 2019. \[11\] C. Zheng et al., “A novel equivalent model of active distribution networks based on LSTM,” IEEE Trans. Neural Netw. Learn. Syst., vol. 30, no. 9, pp. 2611–2624, Sep. 2019. \[12\] H. Bai, P. Zhang, and V. Ajjarapu, “A novel parameter identification approach via hybrid learning for aggregate load modeling,” IEEE Trans. Power Syst., vol. 24, no. 3, pp. 1145–1154, Aug. 2009. \[13\] P. Regulski, D. S. Vilchis-Rodriguez, S. Djurovic, and V. Terzija, “Estimation of composite load model parameters using an improved particle swarm optimization method,” IEEE Trans. Power Del., vol. 30, no. 2, pp. 553–560, Apr. 2015. \[14\] X. Wang, Y. Wang, D. Shi, J. Wang, and Z. Wang, “Two-stage WECC composite load modeling: A double deep Q-learning networks approach,” IEEE Trans. Smart Grid, vol. 11, no. 5, pp. 4331–4344, Sep. 2020. \[15\] S. Afrasiabi et al., “Wide-area composite load parameter identification based on multi-residual deep neural network,” IEEE Trans. Neural Netw. Learn. Syst., vol. 34, no. 9, pp. 6121–6131, Sep. 2023. \[16\] B. Tan, J. Zhao, and N. Duan, “Amortized Bayesian parameter estimation approach for WECC composite load model,” IEEE Trans. Power Syst., vol. 39, no. 1, pp. 1517–1529, Jan. 2024. \[17\] J. Bialek, “What does the GB power outage on 9 August 2019 tell us about the current state of decarbonised power systems?” Energy Policy, vol. 146, Art. no. 111821, Nov. 2020. \[18\] Y. Wang, C. Lu, P. Wu, et al., “Online realization of an ambient signal based load modeling algorithm and its application in field measurement data,” IEEE Trans. Ind. Electron., vol. 69, no. 7, pp. 7451–7460, Jul. 2022. \[19\] C. Qi and X. Wang, “Comparative study of data-driven dynamic load model identification methods based on simulated and actual PMU data,” in Proc. North Amer. Power Symp., Salt Lake City, UT, USA, 2022, pp. 1–6. \[20\] X. Zhang, D. J. Hill, C. Lu, and Y. Song, “A hierarchical framework for ambient signals based load modeling: Exploring the hidden quasi convexity,” IEEE Trans. Power Syst., vol. 36, no. 6, pp. 5780–5791, Nov. 2021. \[21\] Y. Gu and T. C. Green, “Power system stability with a high penetration of inverter-based resources,” Proc. IEEE, vol. 111, no. 7, pp. 832–853, Jul. 2023. \[22\] J. Caballero-Pena, C. Cadena-Zarate, A. Parrado-Duque, and G. Osma- ˜ Pinto, “Distributed energy resources on distribution networks: A system atic review of modelling, simulation, metrics, and impacts,” Int. J. Electr. Power Energy Syst., vol. 138, Art. no. 107900, Jun. 2022. \[23\] R. Chakraborty, H. Jain, and G. Seo, “A review of active probing-based system identification techniques with applications in power systems,” Int. J. Electr. Power Energy Syst., vol. 140, Art. no. 108008, Sep. 2022. \[24\] J. W. Pierre, N. Zhou, F. K. Tuffner, J. F. Hauer, D. J. Trudnowski, and W. A. Mittelstadt, “Probing signal design for power system identification,” IEEE Trans. Power Syst., vol. 25, no. 2, pp. 835–843, May 2010. \[25\] S. Boersma, X. Bombois, L. Vanfretti, J. C. Gonzalez-Torres, and A. Benchaib, “Probing signal design for enhanced damping estimation in power networks,” Int. J. Electr. Power Energy Syst., vol. 129, Art. no. 106640, Jul. 2021. \[26\] J. Peng et al., “Probing signal-based inertia and frequency response es timation for power systems with high levels of inverter-based resources,” in Proc. IEEE PES. Gen Meeting, Seattle, WA, USA, 2024, pp. 1–5. \[27\] M. Zhang, Q. Xu, and X. Wang, “Physics-informed neural network based online impedance identification of voltage source converters,” IEEE Trans. Ind. Electron., vol. 70, no. 4, pp. 3717–3728, Apr. 2023. \[28\] M. Mukherjee, S. V. Singh, A. M. Kulkarni, K. Dey, and A. M. Gole, “Online estimation of linearized IBR models using ambient noise and external excitation,” Electr. Power Syst. Res., vol. 251, Art. no. 112208, 2026. \[29\] J. Zhang and H. Xu, “Online identification of power system equivalent inertia constant,” IEEE Trans. Ind. Electron., vol. 64, no. 10, pp. 8098– 8107, Oct. 2017. \[30\] J. Xu, B. Xie, S. Liao, Y. Sun, D. Ke, J. Yang, P. Li, L. Yu, Q. Xu, and X. Ma, “Online assessment of conservation voltage reduction effects with micro-perturbation,” IEEE Trans. Smart Grid, vol. 12, no. 3, pp. 2224–2238, May 2021. \[31\] P. Pan, H. Hu, D. Xiao, Y. Song, and Z. He, “An improved controlled frequency-band impedance measurement scheme for railway traction power system,” IEEE Trans. Ind. Electron., vol. 68, no. 3, pp. 2184– 2195, Mar. 2021. \[32\] H. Fang, X. Zhang, Y. Wang, and C. Lu, “Probing-based load modeling and optimal low-level signal design,” IEEE Trans. Power Syst., early access, 2026, doi: 10.1109/TPWRS.2026.3673414. \[33\] R. Storn and K. Price, “Differential evolution–A simple and efficient heuristic for global optimization over continuous spaces,” J. Global Optim., vol. 11, no. 4, pp. 341–359, Dec. 1997.

