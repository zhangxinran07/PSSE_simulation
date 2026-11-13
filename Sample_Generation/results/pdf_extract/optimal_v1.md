<!-- Page 1 -->

1

# Probing-Based Load Modeling and Optimal Low-Level Signal Design

Haonan Fang, Graduate Student Member, IEEE, Xinran Zhang, Senior Member, IEEE, Ying Wang, Member, IEEE, Chao Lu, Senior Member, IEEE

Abstract—As a crucial aspect of analyzing power system dynamics, load modeling has been continually aimed at increasing the accuracy of models and attaining the ability to track time varying dynamics. In this paper, a probing-based load modeling approach is proposed. This innovative method actively injects probing signals into power systems through controllable devices to better excite load dynamics and has become increasingly ap plicable due to the increasing integration of controllable devices. To investigate this method and leverage its signal controllability advantages, an optimal probing signal design method is further proposed. First, theoretical identification errors are derived based on a reformulated frequency-domain identification model, providing a basis for evaluating identification performance. Subsequently, an optimization model for probing signal design tasks is established to minimize the theoretical variance, and a two-step optimization approach is developed to solve this prob lem. Case studies demonstrate that the optimal probing signals achieve superior identification performance while inducing fewer disturbances, which verifies the effectiveness of the proposed method. Furthermore, the consistency between the theoretical analysis and simulation results also provides additional validation for the proposed identification error model. Index Terms—Load modeling, probing signal design, identifi cation error model, multisine signal

I. INTRODUCTION L OAD modeling has always been a significant focus in power system studies, as load dynamics play a critical role in power system simulations and stability analyses. In extreme scenarios, inaccurate load models may even result in flawed power system stability assessments \[1\], \[2\]. However, establishing accurate load models is challenging since load models represent the aggregated behavior of all the load components at lower voltage levels. These load components are diverse and always changing due to user behavior, making aggregated load dynamics complicated and time-varying. As a significant methodology for establishing load models, the measurement-based approach has continued to attract sus tained research interest. This widely adopted approach refers to identifying the parameters of a preselected load model structure based on measurement data \[3\]. Among the existing methods, the ZIP+IM model has been applied as one of the fundamental load model structures because of its ability to rep resent the combined dynamics of both static and dynamic load models \[4\]–\[6\]. Furthermore, to better capture increasingly diverse and complex load dynamics, the Western Electricity Coordinating Council (WECC) has developed the composite load model with distributed generation (CMPLDWG), which incorporates multiple new load components \[7\], \[8\]. While complex load model structures can represent sophisticated load dynamics, identification becomes challenging. To ad dress these issues, deep learning techniques, such as double

This work was supported in part by the National Science and Technology Major Project of China under Grant 2025ZD0805804. deep Q-learning and evolutionary deep reinforcement learning, have been applied \[9\], \[10\]. Additionally, a combination of deep learning with Bayesian estimation has been explored to achieve load model identification by learning the posterior distributions of parameters \[11\], \[12\]. Recently, the stochastic characteristics of load models in small transient time intervals have been considered \[13\]–\[15\], and corresponding time varying load modeling approaches have been proposed. The prior studies have made great progress toward the de velopment of advanced load models for representing complex load dynamics. However, these methods are conventionally based on infrequent large disturbance events. Consequently, the rarity of such events limits viable load modeling opportuni ties, making it difficult to continuously track time-varying load properties over the long term. To overcome this problem, an ambient signal-based load modeling method was proposed in \[16\], \[17\]. Ambient signals refer to small disturbances caused by daily power system operations, such as load fluctuations and renewable energy variations \[18\]. Since ambient signals always exist in power systems, this method can track time varying load properties. However, the weak intensity and uncertain properties of ambient signals are key limitations. Owing to these weaknesses, only simple load structures are applied in the current studies, and the reliability of the obtained load modeling results remains a concern. It is challenging to consider both the tracking of time varying load properties over the long term and the reliability of identification. Fortunately, the increase in the controllable resources possessed by power systems is creating opportunities to simultaneously satisfy these requirements via probing-based identification. Probing-based identification refers to actively injecting designed probing signals into a system to excite its responses and utilizing resulting response data to identify system models \[19\]. To date, this method has been successfully applied in power system identification, inertia estimation, and impedance measurement tasks \[20\]–\[34\]. Since 2005, the WECC has carried out a series of system identification tests, including those involving brake insertion, single-mode square waves, and low-level pseudorandom noise, to verify the effectiveness of probing-based approaches \[20\], \[21\]. Owing to the controllability of probing signals, diverse signal waveforms can be selected in research. Multi-sine signals are one of the significant options for probing-based approaches \[22\]–\[30\]. In \[22\], a time-domain swapping algorithm is applied to optimize the phase of multi-sine probing signals, thereby reducing the crest factor and improving the signal-to-noise ratio (SNR). Building on this approach, a Chebyshev-based estimation method is adopted in \[24\] to achieve higher algorithm effi ciency and better crest factor reduction performance. In \[25\] and \[26\] , an optimization model for multi-sine signal design is established to minimize probing-induced disturbances to


<!-- Page 2 -->

2

power systems while limiting the estimation error, and its effectiveness is further verified through a real-time simulation prototype in \[27\]. Beyond power system electromechanical mode estimation, multi-sine probing signals also find ap plication in generator parameter identification \[28\], inertia constant estimation \[29\], and inverter-based resource online estimation \[30\]. Meanwhile, other probing signal options are also available. For instance, a Hann signal-based approach has been tested for inertia estimation purposes by using a battery energy storage system as the excitation source \[31\], \[32\]. In impedance measurement studies, chirp signals with time-varying frequencies are common probing signals, as they can effectively cover the frequency range of interest \[33\], \[34\]. While probing-based identification has been proven effec tive in other fields, previous studies on probing-based load modeling methods remain limited due to probing resource constraints and high costs. With the increasing integration of controllable electronic devices, such as renewables, energy storage, and electric vehicles, these limitations are being alle viated. The controllable devices on the load side can flexibly inject probing signals into power systems and excite load dynamics. Therefore, it is possible and promising to apply the load dynamics excited by probing signals in load modeling scenarios to achieve improved accuracy, as long as the probing signals are well designed. To fully exploit the controllability of probing signals while avoiding excessive power system disturbance, designing sig nals with an intensity constraint is a critical aspect of probing based load modeling. This is a novel challenge that cannot be directly addressed by referencing other fields, as it depends on the characteristics of the given load modeling problem. Specifically, the load responses excited by different probing signals exhibit distinct dynamic behaviors and have different identifiability levels, both of which are determined by the load model structure and its parameters. Therefore, the performance of probing signals is intrinsically linked to the associated load model. Consequently, to design targeted low-level probing signals, it is necessary to establish a method for analyzing the identifiability of the excited load responses of specific load models as the theoretical foundation. Furthermore, to bridge the gap between theoretical analyses and practical probing signals, a quantitative evaluation framework and a feasible signal design process are needed. These important tasks constitute the core focus of this paper. We propose an optimal design method for intensity-limited probing signals, which spans the process from theoretical analysis and signal design problem modeling to optimization. The contributions of this paper are summarized as follows. 1) A probing-based load modeling method, which can better track the time-varying properties of load models and improve the reliability of identification while not causing excessive disturbances in power systems, is proposed. The key innovation in this paper is that probing signals are thoroughly analyzed and meticulously designed to enhance the load modeling performance achieved under signal energy constraints. 2) A theoretical identification error is derived based on a re formulated frequency-domain identification model, which reveals the relationships among the identification results, probing signals and noise. This derivation can serve as a criterion for optimally designing probing signals. 3) A probing signal optimization model is established based on minimizing the theoretical identification error and then solved through a two-step optimization method. This two step optimization method involves frequency optimization and amplitude optimization and can exhibit improved computational feasibility. In this way, an optimal prob ing signal with great identification performance can be obtained with theoretical guarantees. The remainder of this paper is organized as follows. Section II introduces the basic concepts and characteristics of probing based load modeling, as well as the multisine signals applied in this paper. In Section III, frequency-domain reformulations of both the ZIP+IM load model and the identification model are presented, serving as the foundation for the subsequent derivation. The theoretical identification error, which is the evaluation metric for probing signals, is derived in Section IV. Then, a probing signal optimization model and a correspond ing two-step optimization method are proposed in Section V. Finally, in Section VI, case studies are conducted to validate the proposed method. Section VII concludes this paper.

II. CONCEPTS OF PROBING-BASED LOAD MODELING

Probing-based load modeling refers to identifying load models based on the dynamic responses excited by probing signals that are actively injected into power systems. As shown in Fig. 1, there are many controllable resources on the load side of a power system. The designed probing signals can be superimposed onto the control signals of these controllable devices to generate specific excitations, inducing fluctuations in the power and voltage signals of load buses, which can then be applied to identify load model parameters. According to the utilized connection method, controllable devices can be classified into series-connected devices and parallel-connected devices, as shown in Fig. 2. Parallel connected probing devices include distributed photovoltaics (PVs), energy storage systems and static synchronous compen sators (STATCOMs). These devices can excite power systems by controlling their injected active or reactive power. Series connected probing devices include power electronic transform ers (PETs), static synchronous series compensators (SSSCs) and the series-connected components of unified power flow controllers (UPFCs). They can be regarded as equivalent voltage sources connected with the power system in series, thereby enabling the direct regulation of the bus voltage. As a novel methodology, this probing-based load modeling method offers distinct advantages. Compared with that of the ambient signal-based approach, the intensity of the probing signals can be deliberately adjusted to a higher level on the premise of not affecting the security of the system so that the load dynamics can be more sufficiently excited. Moreover, the waveform of the probing signals can be controlled to further increase the resulting identification accuracy. In contrast with the large disturbance-based approach, this new load modeling method can be periodically conducted to track time-varying


<!-- Page 3 -->

3

Fig. 1. Implementation of probing-based load modeling

Fig. 2. Probing device connection methods TABLE I COMPARISON AMONG LOAD MODELING METHODS Data source of load modeling Large Disturbances Ambient Signals Probing Signals Data Sources Natural Natural Artificial Signal Uncontrollable Uncontrollable Controllable Controllability Signal Amplitude High Low Low to (Excitation Level) Medium Impacts on Large Minor Minor to Power System Moderate Execution Frequency Occasionally Periodically Periodically Track Time-varying No Yes Yes Load Properties?

load properties, as the timing of its probing signal injection process is controllable. Although the injection of probing sig nals may also cause additional disturbances to power systems, these disturbances are tolerable when the energy of the probing signals is constrained. Table I presents a comparison among these load modeling methods. A simple hypothetical scenario is considered for conducting preliminary research on this method: a load bus disturbed by an external series-connected controllable device. Within this conceptual framework, the device could be a UPFC, which can operate at the 500-kV level and synthetically control the voltage and power flow \[35\]. Alternatively, at a medium voltage level, it could be a PET, which is already available with megawatt-scale capabilities \[36\]. Since these devices are inverter-based, their control schemes can be continuous and fast. Moreover, their direct bus voltage regulation property is beneficial for probing signal design tasks, as the influence of the transmission impedance between the probing devices and the loads can be ignored. In this work, a multisine signal is selected as the probing signal, which is defined as follows:

V (t) = Xm

i=1 V (ωi) sin(ωit + φv(ωi)) (1)

where m is the number of components and V (ωi) and φv(ωi) represent the amplitude and phase of the i th frequency com ponent, respectively. ωi and fi are related by ωi = 2πfi . Compared with probing signals, multisine signals are su perior. First, their waveforms are relatively smooth, unlike pulse signals, which mitigates the potential damage suffered by probing devices. In addition, they can conveniently cover all frequency ranges of interest. While the basic waveform of the probing signals is des ignated, the optimal frequencies and amplitudes still require determination, which is the focus of the subsequent sections.

III. FREQUENCY-DOMAIN REFORMULATION OF THE LOAD MODELING PROBLEM

In this section, several concepts related to load modeling are revisited. Furthermore, a frequency-domain reformulation of the load modeling problem is conducted, which serves as the foundation for the subsequent derivations.

A. Load Model and Its Frequency-Domain Expression 1) Composite Load Model Structure: A widely used composite load model is applied in this paper; it consists of a ZIP model and a three-order motor model. Moreover, the state–space formulae of the motor model can be transformed into a simplified form \[9\], as shown below:  

 dFd dt = −bFd + sω0Fq + aVd dFq dt = −bFq − sω0Fd + aVq ds dt = 1 H2 (Tm − VqFd + VdFq) (2)

where a = 1 Td0X′ ( X X′ − 1), b = X Td0X′ , Fd = Ed/X′ , Fq = Eq/X′ . X, X′ are the open circuit reactance and transient reactance of the rotor, respectively; Td0 is the open-circuit time constant of the rotor; H2 is the inertia time constant; and Tm is the mechanical torque. Vd, Vq are the d-axis and q-axis components of the voltage phasor, respectively, and Ed, Eq are the d-axis and q-axis components of the electromotive force phasor, respectively. s is the slip ratio. The output of the load model can be given as follows: ( P = FdVq − FqVd + PzV 2 + PiV + Pp

Q = −VdFd − VqFq + QzV 2 + QiV + Qp (3)

The composite load model has a total of 10 parameters, i.e., \[a, b, H2, Tm, Pz, Pi , Pp, Qz, Qi , Qz\].

<!-- Image omitted -->

<!-- Image omitted -->


<!-- Page 4 -->

4

2) Frequency-Domain Model of a Linearized Load: As probing signals are expected not to severely disturb power systems, assuming that the load model is approximately linear for subsequent analysis purposes is reasonable. The linearized load model in this paper is an improvement over that of \[37\]. Specifically, formulae (2) can be written in an equivalent polar coordinate form as follows:  

 dF dt = aV cos δ − bF dδ dt = − aV sin δ F + sω0 + dϕ dt ds dt = 1 H2 (Tm − F V sin δ) (4)

Here,F and V are the magnitudes of the corresponding pha sors, ϕ is the phase angle of the input voltage, i.e., ∠V˙ , and δ is the phase angle difference between the input voltage and the electromotive force, i.e., ∠V˙ − ∠E˙ . The nonlinear terms in (4) are then linearized as:  

 ∆(V cos δ) ≈ −V0 sin δ0∆δ + ∆V cos δ0 ∆(V sin δ) ≈ V0 cos δ0∆δ + ∆V sin δ0 ∆(V sin δ F ) ≈ −V0 sin δ0 F 2 0 ∆F + V0 cos δ0 F0 ∆δ + sin δ0 F0 ∆V (5)

Meanwhile, the steady–state condition can be obtained by setting equation (4) to zero, as follows:  

 cos δ0 = bF0/aV0 sin δ0 = s0ω0F0/aV0 P0 − Ps0 = F0V0 sin δ0 Q0 − Qs0 = −F0V0 sin δ0 (6)

All the above variables with subscripts of 0 represent the steady–state values of the corresponding variables. Ps0 and Qs0 refer to the steady–state active and reactive powers of the static part, respectively. Finally, based on (4)-(6), the state–space equations of the linearized load model are derived as follows: ( x˙ = Ax + Bu y = Cx + Du (7)

where the state, output, and input variables are defined as x = \[∆F/F0, ∆δ, ∆s\] T , y = \[∆P, ∆Q\] T , u = \[∆V /V0, dϕ/dt\] T , respectively. The matrix parameters are expressed as follows:

A = 

 −b −s0ω0 0 s0ω0 −b ω0 −s0ω0F 2 0 /aH2 −bF2 0 /aH2 0 

 (8)

B = 

 b 0 −s0ω0 1 −(P0 − Ps0)/H2 0 

 (9)

C =  P0 − Ps0 −(Q0 − Qs0) 0 Q0 − Qs0 P0 − Ps0 0  (10)

D =  P0 + PzV0 2 − Pp 0 Q0 + QzV0 2 − Qp 0  (11)

Then, the transfer functions of the load model are as follows:

G(s) = C(sI − A) −1B + D =  GV P L (s) G ϕP L (s) G V Q L (s) G ϕQ L (s)  (12)

where GV P L (s), GϕP L (s), GV Q L (s), GϕQ L (s) are the transfer functions from the voltage amplitude and phase angle to the active and reactive powers, respectively. The amplitude and phase angle of the voltage phasor are independent inputs, and the multisine signal design in this paper refers only to the design of voltage amplitude signals. Therefore, only two transfer functions, GV P L (s) and G V Q L (s), must be considered. In addition, the subsequent transfer func tion property analysis is applicable for both GV P L (s) and G V Q L (s), so we use GL(s) to denote them uniformly. 3) Discussion on Errors of Linearized Load Model: The linearization error stems from the approximation in (5), which neglects the effects of higher-order terms. Since the contribution of higher-order terms to this error generally decreases as their order increases, it is reasonable to take the second-order term as a key reference for the qualitative analysis of the linearization error. If the second-order terms in the state-space equations are retained, the equations can be expressed as: x˙ = Ax + Bu + R(x, u) (13)

where the second-order term error R(x, u) is given as:

R(x, u) =  1 2 ν T R1ν 1 2 ν T R2ν F 2 0 2aH2 ν T R3ν T (14)

where ν = \[∆F/F0, ∆δ, ∆V /V0\] T and R1, R2, R3 are:

R1 = 

 0 0 0 0 −b −s0ω0 0 −s0ω0 0 



R2 = 

 −2s0ω0 b s0ω0 b s0ω0 −b s0ω0 −b 0 



R3 = 

 0 −b −s0ω0 −b s0ω0 −b −s0ω0 −b 0 

 (15)

Some state and input variables are not included in ν because they do not directly introduce higher-order term errors in the polar coordinate form. It can be observed that the coefficients of R1, R2, R3 are similar to those in A and B. which means these coefficients are not the core factors determining the relative size of R(x, u) and Ax + Bu. Hence, our focus should be ννT and x, u. Since both ∆F/F0 and ∆V /V0 are normalized, they are generally much smaller than 1 in perturbations, leading to ννT being much smaller than x. Consequently, the error of the linearized model is small and can be acceptable in the analysis, especially when ∆V /V0 remains small. Furthermore, to illustrate the acceptability of the linearized model, Fig. 3 presents the average output error between the nonlinear model and the linearized model under different s and continuous voltage perturbation intensities |∆V |/V0. The output error is defined by:  

 εP = |dPlin(t) − dP(t)| dP(t) × 100%

εQ = |dQlin(t) − dQ(t)| dQ(t) × 100% (16)

where Plin(t) and Qlin(t) are the outputs of the linearized model, while P(t) and Q(t) are that of the original model. Consistent with the preceding analysis, |∆V |/V0 has a more pronounced impact on the linearization error than s. However, even when |∆V |/V0 reaches 0.1, which is already a


<!-- Page 5 -->

5

Fig. 3. Output errors between nonlinear and linearized model significant perturbation, the output error remains below 10%. This indicates that this load model in polar coordinates allows for linearization over a relatively broad range of |∆V |/V0, thereby proving the feasibility of its use in subsequent studies.

B. Identification Model and Its Frequency-Domain Expression Parameter identification refers to finding an optimal set of parameters for minimizing the deviation between a model predicted output and the corresponding measured output. Therefore, the objective function can be defined as follows: min θ X

t ((Pp(t, θ)−Pm(t))2+(Qp(t, θ)−Qm(t))2 ) (17)

where θ is the set of load parameters to be identified. Pp(t, θ), Qp(t, θ) represents the predicted power determined by (3), and Pm(t), Qm(t) represents the measured power. For convenience of analysis, the time-domain identification model is transformed into a frequency-domain model. Based on the Fourier series theory, the power output in the frequency domain Y˙ (ωi) is expressed as shown below: Y˙ (ωi) = |Y˙ (ωi)|e jφ(ωi) = F(Y (t)), Y : P or Q (18)

where Y is a generic symbol for the load output, either P or Q. φ(ωi) refers to the phase angle of Y˙ (ωi). The relationship between Y˙ (ωi) and Y (t) is given as Y (t) = X |Y˙ (ωi)|sin(ωit + φ(ωi)) (19) Moreover, over a complete cycle, the average product of trigonometric functions is given as

sin(ωit + φi) sin(ωj t + φj ) = ( 1/2, i = j 0, i ̸= j (20)

Therefore, the objective function (17) can be derived as follows: min X

t (Yp(t, θ) − Ym(t))2

= min X

t X

i |Y˙ e(ωi)|sin(ωit + φe) !2

= min L 2 X

i |Y˙ e(ωi)| 2 (21)

where Yp(t, θ) and Ym(t) represent the predicted and mea sured outputs in the time-domain, i.e., Pp(t, θ)/Qp(t, θ) and Pm(t)/Qm(t), respectively, while Y˙ p(ωi) and Y˙m(ωi) are their respective frequency-domain transformations. L is the data length, φe is the phase angle of Y˙ e(ωi), which is defined as:

Y˙ e(ωi) = |Ye(ωi)|e jφe = Y˙ p(ωi) − Y˙m(ωi) (22) Therefore, the total error is equivalent to the summation of the respective errors induced across all frequency components. IV. IDENTIFICATION ACCURACY ACHIEVED UNDER PROBING SIGNALS: AN IDENTIFICATION ERROR MODEL In this section, an identification error model is derived to reveal the impacts of probing signals and noise on identifica tion errors. Based on this model, the performance of probing signals in identification tasks can be quantitatively evaluated.

Fig. 4. Transfer function from a voltage input to a measurement power output As illustrated in Fig. 4, the measurement power output, with noise accounted for, can be expressed as shown below:

Y˙m(ωi) = GL(jωi , θr)V˙ (ωi) + N˙ (ωi) (23) Moreover, the model output in the frequency domain can be given as follows: Y˙ p(ωi) = GL(jωi , θiden)V˙ (ωi) (24) where θr and θiden are real and identified parameters, re spectively, N˙ (ωi) is the noise, V˙ (ωi) is the input voltage, and GL(jωi) is the transfer function of the load model. Therefore, Y˙ e(ωi) in (22) can also be written as Y˙ e(ωi)= (GL(jωi ,θiden)−GL(jωi ,θr))V˙ (ωi)−N˙ (ωi) (25) All of the above variables are complex numbers, and their decomposition into radial and lateral components relative to GL(jωi , θr)V˙ (ωi) is illustrated in Fig. 5.

Fig. 5. Decomposition of the frequency-domain variables In Fig. 5, NR(ωi) and NL(ωi) are the decomposition com ponents of N˙ (ωi). GV R(ωi) and GV L(ωi) are the decompo sition components of (GL(jωi , θiden) − GL(jωi , θr)) V˙ (ωi). The magnitude of Y˙ e(ωi) can be derived as follows: |Ye(ωi)| 2 =|NR(ωi)−GVR(ωi)| 2+|NL(ωi)−GVL(ωi)| 2 (26) Since the expression of GL(jω, θ) has been given in (12), the difference between GL(jωi , θr) and GL(jωi , θiden) can be described by their partial derivatives. Specifically, GV  R(ωi) and GV L(ωi) can be given as 

 GVR(ωi) = X

j V (ωi) ∂K(ωi , θr) ∂θj ∆θj

GVL(ωi) = X

j V (ωi) ∂φG(ωi , θr) ∂θj K(ωi , θr)∆θj (27)

where K(ωi ,θr)=|GL(jωi ,θr)|, φG(ωi ,θr)=∠GL(jωi ,θr) (28) From the preceding derivations in (23)-(27), it follows that the frequency-domain identification model (21) constitutes a linear least-squares problem, which is given as shown below: min ||Z − J∆θ|| (29)

<!-- Image omitted -->

<!-- Image omitted -->

<!-- Image omitted -->


<!-- Page 6 -->

6

Zp/q T =  NR(ω1) NL(ω1) · · · NR(ωm) NL(ωm) 

Jp/q = 

             V (ω1) ∂K(ω1, θr) ∂θ1 V (ω1) ∂φG(ω1, θr) ∂θ1 K(ω1, θr)

- · ·

V (ω1) ∂K(ω1, θr) ∂θn V (ω1) ∂φG(ω1, θr) ∂θn K(ω1, θr)

. . . V (ωm) ∂K(ωm, θr) ∂θ1 V (ωm) ∂φG(ωm, θr) ∂θ1 K(ωm, θr)

- · ·

V (ωm) ∂K(ωm, θr) ∂θn V (ωm) ∂φG(ωm, θr) ∂θn K(ωm, θr) 

             (31)

Note that the load model outputs include P and Q, which are both represented by Y in the deduction process. Therefore, Z and J should be formed by concatenating their respective matrices as follows: Z T = \[Zp T , Zq T \], J T = \[Jp T , Jq T \] (30)

where the definitions of Zp/q and Jp/q are given in (31). Here, the subscripts p and q denote the channels for active power and reactive power outputs, respectively. The solution of a linear least-squares problem is given as shown below: ∆θ = (J T J) −1 J T Z (32)

Consequently, the variance of ∆θ is directly obtained as V ar(∆θ) = V ar(HZ) = (H ◦ H)V ar(Z) (33)

where H = (J T J) −1 J T , and the symbol ◦ denotes the Hadamard product operation (the elementwise multiplication of two matrices with the same dimensions). In (33), Z ∈ R 4m×1 , J ∈ R 4m×n, and H ∈ R n×4m, where n is the number of parameters and m is the number of frequencies. Since Z represents the noise distribution and H contains information about the probing signals, this theoretical identification model reveals the effects of probing signals and noise on the resulting identification performance.

V. TWO-STAGE OPTIMIZATION METHOD FOR PROBING SIGNAL DESIGN A. Optimization Model for Probing Signal Design The performance of probing signals in an identification problem can be evaluated by V ar(∆θ). Therefore, the objec tive of probing signal design can be achieved by minimizing this identification variance, which is expressed as follows: min W · V ar(∆θ) = min (WH ◦ WH)V ar(Z) (34) where W ∈ R 1×n, reflecting the relative importance of different parameters. The noise distribution vector Z can be adapted according to specific conditions to enhance the applicability of the proposed method. The decision variables of this optimization model, which can further determine H, are the probing signal frequencies and their corresponding voltage amplitudes:  

 Fopt = \[f1, f2, ...fm\] V opt = \[V (ω1), V (ω2), ..., V (ωm)\]

ωi = 2πfi , i = 1, 2, ..., m (35)

In this optimization problem, the number of frequencies m is fixed to reduce the difficulty of the optimization computa tion. This is reasonable because when m is sufficiently large, the signal energy can still be reallocated to the core frequencies during optimization and will not affect the final results. Owing to the requirement of keeping the amplitudes of the probing signals small, the optimization problem is constrained by the signal energy limit, which is given as shown below:

Signal Energy ≡ Xm

i=1 V (ωi) 2 < Ethre (36)

where Ethre is the signal energy threshold. Finally, the optimization model for probing signal design purposes is obtained as min F opt,V opt (WH ◦ WH)V ar(Z)

s.t.  

 Fopt = \[f1, f2, ...fm\] V opt = \[V (ω1), V (ω2), ..., V (ωm)\]

ωi = 2πfi , i = 1, 2, ..., m Xm

i=1 V (ωi) 2 ≤ Ethre

fmin ≤ fi−1 < fi ≤ fmax, i = 2, 3, ...m 0 ≤ V (ωi) ≤ Vmax, i = 1, 2, ...m (37)

where fmin and fmax represent the upper and lower frequency limits, respectively, and Vmax denotes the upper voltage limit.

B. Probing Signal Design Process In the optimization model given in (37), both the frequencies and amplitudes are decision variables. However, optimizing them simultaneously is challenging. Since the frequencies determine the major basic properties of the excited load dynamics, the identification performance is expected to im prove once the core frequencies are discovered. Therefore, we propose a two-step frequency–voltage optimization method. 1) Step 1: Core Frequency Selection: The first step is aimed at selecting the core frequencies that have significant impacts on the resulting identification performance. Assuming uniform voltage amplitudes for all frequencies, the optimization model is given as follows: min F opt (WH ◦ WH)V ar(Z) + α(∆fthre − ∆fmin)

s.t.  

 Fopt = \[f1, f2, ...fm\] fi ∈ F, i = 1, 2, ..., m

Vi = Ethre √ m , i = 1, 2, ..., m ∆fmin =min {∆fthre, |fi−fj |} , i, j = 1, 2, ...m (38)

where F is a discrete set of selectable frequencies and α is a penalty factor for preventing overly small frequency intervals.


<!-- Page 7 -->

7

Apparently, the partial derivatives (27) for close frequencies are similar, which implies that their impacts on (31) are similar. Therefore, performing optimization over a continuous frequency interval is unnecessary, and the optimal frequencies are selected from a discrete frequency set. One advantage of applying the discrete set is that it can reduce the computational complexity of the method, as the partial derivative vectors of each frequency can be calculated in advance and directly used during the optimization process. In addition, repetitive frequencies are undesirable for the amplitude optimization step conducted next. Therefore, a penalty term α is used in (38). 2) Step 2: Amplitude Optimization: Next, amplitude optimization is conducted based on the op timal frequencies Fopt derived from frequency optimization. This step is aimed at reallocating the signal energy at each frequency, and the optimization model can be given as follows: min V opt (WH ◦ WH)V ar(Z)

s.t.  

 V opt = \[V (ω1), V (ω2), ..., V (ωm)\]

Xm

i=1 V (ωi) 2 = Ethre, i = 1, 2, ...m

0 ≤ V (ωi) ≤ Vmax, i = 1, 2, ...m fi = Fopt,i, i = 1, 2, ...m (39)

where Fopt,i is the i th element of Fopt.

C. Optimization Algorithm Solving optimization problems requires optimization algo rithms. Considering that the proposed optimization model involves the computation of numerous complex partial deriva tives, a globally optimal optimization algorithm with a simple implementation, i.e., the differential evolution (DE) method, is applied in this paper. DE is a heuristic algorithm whose process involves population initialization, mutation, crossover, and selection. There are several advantages to choosing the DE algorithm. First, it is a population-based, multi-start algorithm, which allows for exploring a broader search space, while also ensuring low sensitivity to the selection of initial points. In addition, DE does not require gradient information, making it convenient for handling complex problems. Since this method are not the primary concern of this paper, further details about DE can be found in \[38\].

VI. CASE STUDIES A. Test Environment Settings To validate the effectiveness of the proposed method, simu lations and load identification are conducted on a test system, as shown in Fig. 6. This system is based on a standard benchmark system, i.e., the IEEE 39-bus system \[39\], with a minor modification: the target load model is connected to Bus 15 via a controllable transformer to allow for secondary voltage control, instead of being directly connected. The load model at Bus 40, which is to be identified, is a ZIP+IM model whose parameters are listed in Table II. The simulation is conducted in the Power System Analysis Toolbox (PSAT) in MATLAB, and the hardware used for case studies is a laptop Fig. 6. Test system with adjustable voltage

with an AMD Ryzen 7 (R7 4800H) CPU at 2.9 GHz and 32.0 GB of RAM. The simulation time step size is 0.01 s.

TABLE II LOAD MODEL PARAMETERS X(p.u.) X′ (p.u.) Td0(p.u.) H2(s) Tm(p.u.) Pz(p.u.) 0.9533 0.2523 0.1700 2.4778 1.0 0.3 Pi(p.u.) Pp(p.u.) Qz(p.u.) Qi(p.u.) Qp(p.u.) 0 0.3 0.3 0 0.3

Fig. 7. Flowchart of probing-based load modeling The flowchart of the full process for the probing-based load modeling approach in the case studies is shown in Fig. 7, which consists of three procedures. Procedure 1 is probing signal design, a process of executing the optimization algorithm, in which an optimization model for the probing

<!-- Image omitted -->

<!-- Image omitted -->


<!-- Page 8 -->

8

signals is first constructed and subsequently solved using a two-stage method to determine the probing signals. Procedure 2 is load dynamics excitation, in which the pre viously designed probing signals are injected into the control signal of the controllable transformer to adjust the voltage on the load side, thereby exciting the load dynamics. Procedure 3 is parameter identification, which constitutes the ultimate goal and final step of the entire process. In this phase, the load model parameters are estimated by solving the identification problem formulated based on the obtained load response data (V, P, Q). Since this step follows the same fun damental logic as traditional measurement-based approaches, the methodologies of prior research are fully applicable here. Specifically, this paper adopts a hierarchical identification framework, with further details available in \[17\]. Notably, the original ZIP+IM model remains the ultimate estimation target of the parameter identification procedure, despite probing signal design utilizing the linearized load model as its analytical basis. Besides, the parameter set θ used in the theoretical analysis is limited to \[a, b, H2, Tm, Pz, Qz\], with \[Pp, Qp, Pi , Qi \] excluded. This is because this reduced set can maintain analytical precision while reducing the incurred computational complexity. Pp and Qp are excluded from the calculation of (33). This is because the column corresponding to component P in (31) contains nonzero elements only at f=0. Consequently, removing both the column corresponding to P and the f=0 row from matrix (31) leaves the theoretical variance V ar(∆θ) of the remaining parameters unchanged. Pi and Qi are also excluded from the calculation of (33). This exclusion is necessary because the column corresponding to component I in (31) is linearly dependent on that corre sponding to component Z. As a result, J T J becomes singular when both columns are included in (31). This fact also reveals a dependency between Z and I under small disturbances \[17\]. However, this internal dependency within the ZIP model has almost no effect on the identification of the other parameters, which has been demonstrated in \[40\].

B. Stochastic Load Fluctuations: Analysis and Configuration To solve the optimization model in (37), V ar(Z) should be determined. This parameter characterizes the spectrum of the noise superimposed on the effective load responses. During normal operations, the load power exhibits inherent stochastic fluctuations. These fluctuations have different properties from those of the load responses excited by probing signals and act as interference in probing-based load modeling cases. Thus, they are treated as noise and described by V ar(Z) in the identification error analysis. In this section, the spectra of measurement data are analyzed to construct the simulation noise model. An example of the measurement data from a substation, sampled at 50 Hz, is shown in Fig. 8. To characterize the properties of the measurement data, single-day data from two substations is utilized. For each hourly interval of these data, a 60-second segment with stable fluctuations is selected, and its corresponding spectrum is calculated using the Fourier transform. The resulting curves are presented in Fig. 9. Fig. 8. Measurement data of a substation

Fig. 9. Spectrum of the measured power data

Qualitative observation reveals that all spectra share a similar trend, with the noise energy concentrated in the low frequency range, primarily below 2 Hz. Furthermore, to quan tify the similarities among the spectra, the Pearson correlation coefficient is adopted as a similarity index (SI), defined as:

SI = P i (As1,i − As1)(As2,i − As2)

rP i (As1,i − As1) 2 P j (As2,j − As2) 2 (40)

where As1,i is the spectral amplitude of the i th frequency com ponent in the first spectrum, while As2,i is the corresponding amplitude in the second spectrum. As1 and As2 stand for the average values of the two spectra, respectively. SI close to 1 means the two spectral shapes have high similarity. Table III presents the average similarity index (SI) between a substation’s own spectra at different times, as well as that between the spectra of the two substations. Moreover, the empirical cumulative distribution function (CDF) of the SIs is provided in Fig. 10. It can be found that the spectra from the same substation at different times are more similar than those between different substations, as they not only have a higher SI, but also have over 70% of their SIs greater than 0.8. However, this does not imply that the spectral differences between substations are substantial. For context, the SI between white noise and these spectra is close to 0, considerably lower than the inter-substation SI. Therefore, it can be concluded that the spectra of measured data have distinct and consistent characteristics, thereby supporting the rationality of using a simulation noise model to represent typical real-world conditions. TABLE III AVERAGE SIMILARITY INDEX (SI) OF SPECTRA SI of Substation 1 SI of Substation 2 SI between Substation 1 & 2 P spectra 0.8608 0.8597 0.7356 Q spectra 0.8763 0.9162 0.7233

<!-- Image omitted -->

<!-- Image omitted -->


<!-- Page 9 -->

9

Fig. 10. Empirical CDF of Spectral SIs To simulate this actual condition while accounting for implementation convenience, the simulation noise model is designed as low-pass filtered white noise: 60% with a 0.2- Hz cutoff frequency, 20% with a 1.5-Hz cutoff frequency, and 20% with a 15-Hz cutoff frequency. Specifically, the average similarity indices (SIs) between the spectrum of this noise model and the active/reactive power spectra of Substation 1 can reach 0.8208 and 0.8276, respectively, while those for Substation 2 are 0.8134 and 0.7713. These values indicate a high degree of similarity, demonstrating the effectiveness of the proposed noise model in representing realistic conditions. Finally, we determine V ar(Z) by sampling from the vari ance of the simulated noise described above. In V ar(Z), N˙ (ωi) is decomposed into NR(ωi) and NL(ωi). Since the noise is independent of the real load responses, the properties of NR(ωi) and NL(ωi) should be similar. Thus, it is reason able to assume the following: V ar(NR(ωi)) = V ar(NL(ωi)) ∝ V ar(N(ωi)) (41) Since the optimization results only depend on the relative magnitudes of the elements in V ar(Z), we normalize V ar(Z) by using its value at 10 Hz as the baseline, thereby avoiding the need to handle conversion factors. The resulting profile of V ar(Z) is shown in Fig. 11.

Fig. 11. Profile of V ar(Z)

C. Probing Signal Design and Performance Validation 1) Determination of Frequency Set Range: The reasonable range of the frequency set F in probing signal design is determined in this subsection. In the identification error model (33), J reflects the effect of load model parameters on the identification error. It shares a similar physical interpretation with the parameter sensitivity, as larger elements indicate the output is more sensitive to parameter changes. Observations reveal that J is composed of submatrices associated with individual frequency components, denoted as J(ω), which is defined as: J(ω) T =  Jp(ω) T , Jq(ω) T 

Jp/q(ω) = V (ω) " ∂K ∂θ1

- · ·

∂K ∂θn ∂φG ∂θ1 K · · · ∂φG ∂θn K # (42) where K = K(ω, θr), φG = φG(ω, θr). Therefore, by analyzing the submatrix J(ω), the effect of different frequency components on the identification results can be preliminarily characterized, thereby guiding the selec tion of frequency set range. Specifically, the impact of a fre quency component ω is characterized by the Frobenius norm of its J(ω), denoted as ∥J(ω)∥F . Meanwhile, the difference in the contributions between two frequencies is quantified by the distance between their corresponding submatrices, which is also based on the Frobenius norm and defined as follows: dis(ωi , ωj ) = ∥J(ωi) − J(ωj )∥F (43) After obtaining the relative distances between different frequencies, two-dimensional visualization can be generated by the multidimensional scaling (MDS) algorithm, as shown in Fig. 12. Meanwhile, the color intensity represents the ratio of ∥J(ω)∥F for each frequency to the maximum ∥J(ω)∥F . It can be observed that for frequencies above 10 Hz, the relative positions of the corresponding points are very close to each other, indicating that the impacts of these frequencies in identification are similar. Therefore, it is reasonable to define the set F as a set of frequency components from 0.1 Hz to 10 Hz, at an interval of 0.1 Hz. Notably, 10 Hz is not a strictly derived boundary but a reasonable choice, as any frequency near this value could be justifiable. In addition, m is set to 10.

Fig. 12. Visualization of Relative Positions of J(ω) 2) Optimal Probing Signal Calculation: After the optimization model parameters are determined, the two-step optimization method is applied to obtain optimized probing signals, the results of which are shown in Table IV. Here, Group 1 is the comparison group, Group 2 shows the results obtained after performing frequency optimization with α set to 0.5, and Group 3 presents the probing signal acquired after implementing voltage optimization. Obj refers to the objective function value, i.e. (WH◦WH)V ar(Z). The curve of the optimal probing signal is illustrated in Fig. 13. TABLE IV PROBING SIGNAL OPTIMIZATION Group Probing Signal Setting Obj

1 f : \[1, 2, 3, 4, 5, 6, 7, 8, 9, 10\] 3.20 V : \[1, 1, 1, 1, 1, 1, 1, 1, 1, 1\]

2 f : \[2.0, 2.3, 2.7, 6.9, 8, 8.3, 8.6, 8.9, 9.2, 9.6\] 1.41 V : \[1, 1, 1, 1, 1, 1, 1, 1, 1, 1\]

3 f : \[2.0, 2.3, 2.7, 6.9, 8, 8.3, 8.6, 8.9, 9.2, 9.6\] 1.32 V : \[0.157, 0, 0.113, 0, 0, 0, 0, 0.2504, 0, 0\]

<!-- Image omitted -->

<!-- Image omitted -->

<!-- Image omitted -->


<!-- Page 10 -->

10

Compared with that of the uniformly distributed frequencies (Group 1), the Obj of the optimized frequencies (Group 2) significantly decreases, which implies a significant load iden tification improvement achieved via frequency optimization.

Fig. 13. Optimal probing signal voltage Moreover, the frequencies in Group 2 are concentrated around several specific values, including 2 Hz, 2.7 Hz, and 9 Hz. This is because the impacts of different frequencies on load identification are different. On the other hand, the results are not confined to a single frequency; thus, sufficiently excited dynamics can be achieved. The frequencies of the optimal probing signals and high noise area in the spectrum are disjoint, as shown in Fig. 13. This ensures that the noise does not significantly interfere with the information-carrying frequencies. Therefore, it can be concluded that probing signals are expected to be injected into the frequency band with lower noise levels. TABLE V ALGORITHMIC EFFICIENCY COMPARISON

Condition 1 (100 Cases) Condition 2 (100 Cases) Condition 3 (100 Cases) Step 1 Step 2 Min Obj 1.28 1.29 1.41 1.32 Calculation Time 2280.5 s 15.73 s 3.78 s 7.63 s %(Obj ≤ 1.32) 52% 34% - 63%

Furthermore, to verify the rationality of selecting a discrete frequency set and adopting a two-step optimization approach, we compare the computational efficiency levels observed un der three conditions: 1) Condition 1 uses a continuous frequency set from 0.1 Hz to 10 Hz and simultaneously optimizes frequency and amplitude, i.e., solves the optimization model (37). 2) Condition 2 uses a discrete frequency set F and simulta neously optimizes frequency and amplitude, which refers to solving the optimization model (37) with an additional constraint fi ∈ F. 3) Condition 3 uses the discrete frequency set F with two step optimization, specifically, first solving optimization model (38) and then solving model (39). In this context, a single execution of the optimization proce dure to derive an optimal probing signal is termed a case. For each condition, 100 cases are conducted. The computational efficiency results are presented in Table V. The comparison results reveal that selecting a continuous frequency set results in prohibitively high computational costs due to repeated partial derivative calculations. Moreover, the two-stage optimization method requires only approximately Fig. 14. Evolution of Obj across different iterations

70% of the computation time of the single-stage approach. Furthermore, the final Obj values of all three methodologies are very close and obviously lower than that of the initial solution, confirming that the staged process can still maintain strong computational accuracy. Notably, although the single stage method can theoretically achieve better optimal values, the strong nonlinearity of this optimization problem actually makes its solutions unstable. Table V implies that the single stage approach can achieve a lower Obj than the optimal Obj of the two-stage method in only 34% of the cases, whereas the two-stage approach can achieve this feat in 63% of the cases. Therefore, the two-stage optimization approach offers greater practical feasibility. The DE algorithm is applied to solve the aforementioned optimization models. To demonstrate its low sensitivity to initial points, the 100 cases corresponding to the first-step optimization of Condition 3 are analyzed. The comparison of Obj values for the initial solutions and final solutions of these cases is presented in Fig. 15. It can be observed that some initial solutions yield significantly high Obj values. These high Obj values are likely attributable to the near-singular condition of their corresponding J T J matrices. Nevertheless, the DE algorithm successfully optimized all cases to values near the minimum value, confirming its robust global search capability. The impacts of a few poor initial solutions on the final results are minor.

Fig. 15. Distribution of Obj values for initial and final solutions Moreover, the performance of a non-heuristic Sequential Quadratic Programming (SQP) algorithm is compared with that of the heuristic DE method, with the first-step optimiza tion of Condition 3 still serving as the application scenario. The distribution of final Obj values from 100 runs of the SQP algorithm with random initial points is compared against that from the DE algorithm, as shown in Fig. 16. By leveraging gradient information, SQP can achieve a higher optimization efficiency, with an average computation time of only 0.032 s. However, SQP exhibits high sensitivity to the initial points, leading to dispersed final results. Moreover, its solution quality is poor, with both the average and best outcomes being inferior to those obtained by the DE algorithm. For the probing signal design problem, the primary goal is

<!-- Image omitted -->

<!-- Image omitted -->

<!-- Image omitted -->


<!-- Page 11 -->

11

Fig. 16. Distribution of Obj values for different algorithm

to achieve a substantial reduction in the objective function, which corresponds to better performance of probing signals. Convergence to the precise global optimum, while desirable, is not a strict requirement. Therefore, since the DE algorithm consistently produces Obj values close to the minimum, its selection in this study is justified. 3) Simulation and Identification: To further validate the advantages of the optimal probing signal, simulation and load parameter identification tests are conducted based on the following four groups of multisine probing signals: 1) Group 1:random frequencies and random amplitudes; 2) Group 2:uniform frequencies and random amplitudes; 3) Group 3:uniform frequencies and uniform amplitudes; 4) Group 4:optimized frequencies and amplitudes.

TABLE VI THEORETICAL VARIANCE INDICES Group 1 Group 2 Group 3 Group 4 V ar(a/a0) 23.757 9.182 7.524 2.361 V ar(b/b0) 19.031 4.668 5.049 4.068 V ar(H2/H20) 115.97 24.14 22.35 18.20 V ar(Tm/Tm0) 11.58 5.938 4.466 1 V ar(Pz/Pz0) 67.14 23.522 19.09 6.117 V ar(Qz/Qz0) 0.387 0.269 0.267 0.216

Utilizing formula (33), the theoretical variances of the load model parameters for the above four groups are calculated and normalized relative to the V ar(Tm) of Group 4, as presented in Table VI. The indices of all the parameters based on the optimal probing signal are minimized, which implies that the optimal probing signal has the best identification performance. In addition, compared with the indices of parameters \[b, H2\], those of parameters \[a, Tm\] are greater when the optimal probing signal is applied. This finding indicates that their identifiability is improved. The above probing signals are injected into the bus voltage at different signal energy levels. A total of 20 energy levels are set, and they evenly distributed from 4e-7 (p.u.) to 2e-6 (p.u.). Moreover, the noise designed in Section VI-B is added to the load responses, with a standard deviation of 1.5e-3 (p.u.). For each level and signal type, 100 simulation cases are conducted.

Fig. 17. Voltage and power curves of a simulation case The voltage and power curves of a simulation case are shown in Fig. 17. In all the simulation cases, the test system remains stable. When the signal energy is 2e-6 (p.u.), the characteristics of the voltage disturbances are presented in Table VII, where Vpp represents the peak-to-peak magnitude of the voltage fluctuations. First, the largest voltage fluctuations occur at the probing signal injection point, i.e., Bus 40. The voltage fluctuations at other buses are obviously smaller than those at Bus 40. Therefore, system voltages always fluctuate within a safe range, and the probing signals do not cause excessive disturbances in the test system. This fact also implies that since there is room to further increase the energy of the probing signal, there is also potential for further improving the resulting identification accuracy. TABLE VII VOLTAGE DISTURBANCES CHARACTERISTICS IN A SIMULATION Group 1 Group 2 Group 3 Group 4 Vpp of Bus 40 (p.u.) 0.0060 0.0055 0.0062 0.0042 Global Maximum Vpp 2.52e-4 2.72e-4 2.85e-4 1.86e-4 across Bus 1-39 (p.u.) (Bus 15) (Bus 15) (Bus 15) (Bus 15) Average Vpp for 1.12e-4 1.06e-4 1.19e-4 6.42e-5 Bus 1-39 (p.u.)

Fig. 18. Distribution of the identification results

Fig. 19. Variances of the identification results The distribution of the identification results produced when the energy of the probing signal is 1.2e-6 (p.u.) is calculated using kernel density estimation, as illustrated in Fig. 18. First, the identification results are all around the true value, confirming the effectiveness of probing-based load modeling. Notably, the kernel density curve of Group 4 exhibits a sharper peak and a narrower bandwidth, particularly for the parameter \[a, Tm\], which indicates higher identification accuracy. This observation aligns with prior theoretical predictions. Further more, the variance trend of the identification results is shown in Fig. 19. The variance of the identification results obtained for the optimal probing signal consistently remains the lowest among all four signal types, regardless of the signal energy

<!-- Image omitted -->

<!-- Image omitted -->

<!-- Image omitted -->

<!-- Image omitted -->


<!-- Page 12 -->

12

level. Therefore, the optimal probing signal has optimal overall load identification performance. TABLE VIII VARIANCES OF THE IDENTIFICATION RESULTS Group 1 Group 2 Group 3 Group 4 V ar(a/a0) 0.0452 (24.56) 0.0173 (9.43) 0.0119 (6.46) 0.0038 (2.07) V ar(b/b0) 0.0363 (19.74) 0.0079 (4.31) 0.0076 (4.13) 0.0076 (4.13) V ar(H2/H20) 0.2859 (155.52) 0.0512 (27.86) 0.0418 (22.72) 0.0415 (22.57) V ar(Tm/Tm0) 0.0188 (10.25) 0.0108 (5.88) 0.0078 (4.24) 0.0018 (1) V ar(Pz/Pz0) 0.1016 (55.24) 0.0499 (27.16) 0.0399 (21.68) 0.0088 (4.81) V ar(Qz/Qz0) 0.0048 (2.60) 7.63e-4 (0.415) 8.11e-4 (0.441) 5.30e-4 (0.288) The identification results further confirm the validity of our theory. Table VIII lists the variances of the parameters identified for the four probing signals when the energy of the probing signal is 1.2e-6 (p.u.), and the values in parentheses are the normalized variances relative to the V ar(Tm) of Group 4. Comparing Table VI and Table VIII, the normalized variances of the same parameters in the two tables are almost identical. Therefore, our identification error model accurately characterizes the relationships between the variance of the identification results and both the probing signals and noise.

D. Load Modeling Method Comparison The results in the previous sections validated the notion that our approach improves the identification performance of multi sine probing signals. Furthermore, to highlight the advantages of the optimized probing signals over ambient signals or large disturbances, a comparison among the identification results of different load modeling methods is conducted. 1) Comparison with Ambient Signals: In this simulation, the spectrum of ambient signals consists of 60% of a filtered signal with a 0.2-Hz cutoff frequency, 20% of a filtered signal with a 1.5-Hz cutoff frequency, and 20% of a filtered signal with a 15-Hz cutoff frequency, as described in Section VI-B. Moreover, the multisine signal is the same as that used in the previous case studies, i.e., Group 4 in Table IV. To enable a meaningful signal performance comparison, both signals have a consistent signal energy level of 1.2e-6 (p.u.). A total of 100 simulation cases are implemented for each signal to evaluate their identification performance. The similarity between the predicted and measured powers is described by the fitting degree, which is defined as follows:

f it = (1 − s P (yp − ymea) 2

P (ymea − y¯mea) 2 ) (44)

where yp is the predicted power, ymea is the measured power, and y¯mea is the mean value of ymea. A comparison between the dynamics of the measured and predicted powers under the multisine signal and the ambient signal is given in Fig. 20. Compared with those of the ambient signal, the identification results obtained based on the optimal multisine signal have a greater degree of fit. Moreover, as shown in Table IX and Fig. 21, the mean error and dispersion of the identification results based on the optimal multisine Fig. 20. Comparison between the measured and predicted powers. (a) Multisine signal (b) Ambient signal

signal are significantly smaller than those based on the ambient signal. This fact implies that the load dynamics excited by multisine signals are more suitable for identification purposes. TABLE IX IDENTIFICATION RESULTS OF 100 CASES

Actual Scenario 1 Scenario 2 µ(x) (p.u.) Error (%) µ(x)(p.u.) Error (%) a 64.8079 75.3148 16.21 64.7559 0.080 b 22.2303 24.2637 9.147 22.2525 0.100 H2 2.4778 3.0914 24.76 2.5302 2.116 Tm 1 0.9445 5.548 1.0042 0.418 Pp 0.3 0.4023 34.09 0.2980 0.683 Pz 0.3 0.2531 15.62 0.2978 0.723 Qp 0.3 0.3186 6.192 0.3061 2.032 Qz 4.264 4.3678 2.427 4.2670 0.064 Scenario 1: identification under ambient signals. Scenario 2: identification under optimal multisine signals.

Fig. 21. Comparison between the produced identification error distributions

Notably, the ambient signal intensity level is uncontrollable and may be too weak to produce reliable identification results for some load buses. In contrast, controlled probing signals can ensure sufficient signal energy for guaranteeing high load modeling accuracy, thereby filling the load modeling gap for load buses with ambient signal intensities that are too low. 2) Comparison with Large Disturbances: In this subsection, a three-phase short circuit fault is applied on Bus 14 at 1 s and cleared at 1.5 s as the data source of large disturbance-based load modeling. The dynamic curves produced for a case with added noise are shown in Fig. 22.

Fig. 22. Dynamic curves produced under a three-phase short circuit fault The fault intensity is adjusted by changing the ground impedance, with a total of 60 different configurations, so that the correlations between the variance of the identification results and the impacts of faults on the power system are

<!-- Image omitted -->

<!-- Image omitted -->

<!-- Image omitted -->


<!-- Page 13 -->

13

established, as shown in Fig. 23. In this figure, the red markers represent the probing-based load modeling results obtained with a signal energy of 2e-6 (p.u.), whereas the black markers correspond to the large disturbance-based load modeling results produced with an equivalent variance. The large disturbance-based approach requires voltage fluctuations to be induced at levels three or more times greater than those caused by probing signals to attain comparable modeling accuracy. These findings demonstrate that using optimized probing signals can lead to better identification accuracy while causing fewer disturbances, which means less harm to the stability of the power systems.

Fig. 23. Correlation between V ar(∆θ) and Vpp Another important fact is that the faults in these simulation cases are already very small. Typical faults in real power systems tend to be more severe, and they are inherently uncontrollable. Therefore, a key advantage of the probing based approach is that its signal intensity can be controlled to a required low level to avoid disturbing systems too much.

E. Further Discussion The effectiveness and superiority of the proposed probing based load modeling approach have been proven in the above case studies. Although the validation is based on a simple ZIP+IM load model, this result still holds practical significance. However, it must be acknowledged that our approach still faces significant challenges in its application to the more complicated load model structure, such as WECC’s CMPLDWG. Therefore, this subsection will briefly discuss the key questions regarding probing signal design for the complicated load model. 1) Nonlinearity of the complicated load models: Complicated load models have incorporated numerous ad ditional components beyond the ZIP+IM model structure, for example, the DER A model and electronic loads. These new components can operate in different modes under different disturbance conditions. When components show significant nonlinearity, describing the impact of parameters on the re sponse using local sensitivity becomes insufficient. To address this issue, one potential avenue for future work could be designing diversified probing signals for different operational modes. Each stage aims to identify a subset of parameters based on excited local characteristics, thereby reducing the nonlinearity of the model in each stage. 2) Non-continuous influence of several parameters: Several components of the complicated load model, such as the DER A model, contain various nonlinear modules involving limiters, dead bands, and triggering thresholds. The associated protective parameters remain inactive unless the excitation exceeds a certain intensity. Once the excitation is sufficiently strong to activate them, the influence of these parameters tends to saturate. Due to their different features, our approach cannot be applied to designing the probing signal for identifying these parameters. This challenge presents a broader research gap, where even the original multi-sine probing signals may not be optimal for identifying these special parameters, and thus determining the most effective probing signal waveform and its intensity remains an open question for future research. 3) Calculation difficulty for the complicated load model: Complex load models involve a large number of parameters to be identified, which increases the dimension of matrix J. This expansion in the scale of J imposes a higher computa tional burden on the optimal probing signal design. In response to this challenge, two potential pathways are worth investi gating. First, the sensitivity analysis could be leveraged to determine a dominant parameter subset, enabling signal design to focus on these critical parameters. Second, the optimization algorithm itself can be improved to achieve higher efficiency and more robust global search capability. 4) Stability concerns of probing signal injection: In this study, low-level probing signals are utilized, thus stability and security concerns are not prominent. However, identifying more complex load models will generally require stronger probing signals, which raises concerns regarding their potential impact on system security and stability. To account for this factor, future work might involve integrating stability indices into the probing signal design framework, thereby enabling a balance between identification performance and system security. Furthermore, from an implementation perspective, the concept of multiple probing signal injection could be explored to mitigate locally concentrated disturbances and enhance overall operational security.

VII. CONCLUSION

In this paper, a probing-based load modeling method is pro posed. This method is a valuable complement to the traditional load modeling approaches, addressing their blind spots and enabling load modeling to be performed at specific load buses and time intervals where conventional methods may not be applicable. To evaluate the performance of probing signals in identification tasks, the theoretical identification error is derived. An optimal model for probing signal design pur poses, which is aimed at minimizing the theoretical variance of the signal energy constraint, is subsequently established. Moreover, a two-step optimization method involving frequency optimization in the first step and amplitude optimization in the second step is proposed to solve this optimization prob lem. The simulation results indicate that this probing-based approach does not cause excessive disturbances that harm the stability of a power system. Moreover, a case study shows that the optimal probing signal exhibits superior load identification performance, with its variance always being the minimum value. In addition, the consistency between the variance of the identified parameters and their theoretical variance fur ther demonstrates the correctness of our theory. Moreover,

<!-- Image omitted -->


<!-- Page 14 -->

14

### a comparative analysis conducted between the probing-based

### load modeling approach and other methods reveals that well

### designed multisine signals can provide better identification

### accuracy while inducing fewer system disturbances.

### As a novel methodology that can accommodate both time

### varying load property tracking and reliable identification tasks,

### probing-based load modeling shows significant research po

### tential, particularly in the context of the increasingly complex

### load dynamics and enhanced controllability exhibited by mod

### ern power systems. Future work will further focus on probing

### based load modeling, exploring aspects such as probing signal

### designs for more complex load models and the coordination

### of multiple power probing signal injection devices.

REFERENCES

\[1\] Kosterev D. N., Taylor C.W., Mittelstadt W.A., “Model Validation for the August 10, 1996 WSCC system outage”, IEEE Trans. Power Syst., 1999, Vol.14, No.3, pp.967-979, Aug. 1999. \[2\] A. Arif, Z. Wang, J. Wang, B. Mather, H. Bashualdo and D. Zhao, ”Load Modeling—A Review,” IEEE Trans. Smart Grid, vol. 9, no. 6, pp. 5986- 5999, Nov. 2018. \[3\] H. Renmu, M. Jin, and D. J. Hill, “Composite load modeling via measurement approach,” IEEE Trans. Power Syst., vol. 21, no. 2, pp. 663–672, May 2006. \[4\] L. Chavarro-Barrera, S. P ´ erez-Londo ´ no, and J. Mora-Fl ˜ orez, “An adaptive ´ approach for dynamic load modeling in microgrids,” IEEE Trans. Smart Grid, vol. 12, no. 4, pp. 2834–2843, Jul. 2021. \[5\] J. Xie et al., “Imitation and transfer Q-learning-based parameter identifi cation for composite load modeling,” IEEE Trans. Smart Grid, vol. 12, no. 2, pp. 1674–1684, Mar. 2021. \[6\] Y. Lin, Y. Wang, J. Wang, and D. Shi, “Tensor-based parameter reduction of dynamic load models with variable frequency drive,” IEEE Trans. Power Syst., vol. 37, no. 2, pp. 1091-1101, Mar. 2022. \[7\] Z. Ma, Z. Wang, Y. Wang, R. Diao, and D. Shi, “Mathematical represen tation of WECC composite load model,” J. Modern Power Syst. Clean Energy, vol. 8, no. 5, pp. 1015–1023, Sep. 2020. \[8\] H. Huang, Y. Lin, Y. Zhou, Y. Zhao, P. Zhang, and L. Fan, “Data-driven modeling of power system dynamics: Challenges, state of the art, and future work,” iEnergy, vol. 2, no. 3, pp. 200–221, Sep. 2023. \[9\] X. Wang, Y. Wang, D. Shi, J. Wang, and Z. Wang, “Two-stage WECC composite load modeling: A double deep Q-learning networks approach,” IEEE Trans. Smart Grid, vol. 11, no. 5, pp. 4331–4344, Sep. 2020. \[10\] F. Bu, Z. Ma, Y. Yuan, and Z. Wang, “WECC composite load model parameter identification using evolutionary deep reinforcement learning,” IEEE Trans. Smart Grid, vol. 11, no. 6, pp. 5407–5417, Nov. 2020. \[11\] B. Tan, J. Zhao, and N. Duan, “Amortized Bayesian parameter estimation approach for WECC composite load model,” IEEE Trans. Power Syst., vol. 39, no. 1, pp. 1517–1529, Jan. 2024. \[12\] B. Tan, J. Zhao, N. Chiang and N. Duan, ”High-dimension Bayesian parameter estimation for WECC composite load model using realis tic event measurements,” IEEE Trans. Power Syst., early access, doi: 10.1109/TPWRS.2025.3543544. \[13\] M. Cui, J. Wang, Y. Wang, R. Diao, and D. Shi, “Robust time-varying synthesis load modeling in distribution networks considering voltage disturbances,” IEEE Trans. Power Syst., vol. 34, no. 6, pp. 4438–4450, Nov. 2019. \[14\] C. Wang, Z. Wang, J. Wang, and D. Zhao,“Robust time-varying parame ter identification for composite load modeling,” IEEE Trans. Smart Grid, vol. 10, no. 1, pp. 967–979, Jan. 2019. \[15\] M. Khodayar and J. Wang, “Probabilistic time-varying parameter iden tification for load modeling: A deep generative approach,” IEEE Trans. Industr. Inform., vol. 17, no. 3, pp. 1625–1636, 2021. \[16\] C. Qi and X.Wang, “Comparative study of data-driven dynamic load model identification methods based on simulated and actual PMU data,” in Proc. North Amer. Power Symp., Salt Lake City, UT, USA, 2022, pp. 1–6. \[17\] X. Zhang, D. J. Hill, C. Lu, and Y. Song, “A hierarchical framework for ambient signals based load modeling: Exploring the hidden quasi convexity,” IEEE Trans. Power Syst., vol. 36, no. 6, pp. 5780–5791, Nov. 2021. \[18\] J. W. Pierre, D. J. Trudnowski, and M. K. Donnelly, “Initial results in electromechanical mode identification from ambient data,” IEEE Trans. Power Syst., vol. 12, no. 3, pp. 1245–1251, Aug. 1997. \[19\] R. Chakraborty, H. Jain, and G. Seo, “A review of active probing-based system identification techniques with applications in power systems,” Int. J. Elect. Power Energy Syst., vol. 140, 2022, Art. no. 108008. \[20\] N. Zhou, J. W. Pierre, and J. F. Hauer, “Initial results in power system identification from injected probing signals using a subspace method,” IEEE Trans. Power Syst., vol. 21, no. 3, pp. 1296–1302, Aug. 2006. \[21\] D. Trudnowski, D. Kosterev, and J. Wold, “Open-loop PDCI probing tests for the Western North American power system,” in Proc. IEEE PES Gen. Meeting–Conf. & Expo., 2014, pp. 1–5. \[22\] J. W. Pierre, N. Zhou, F. K. Tuffner, J. F. Hauer, D. J. Trudnowski, and W. A. Mittelstadt, “Probing signal design for power system identification,” IEEE Trans. Power Syst., vol. 25, no. 2, pp. 835–843, May 2010. \[23\] J. Zhang, C. Lu, and Y. Han, “MIMO identification of power system with low level probing tests: Applicability comparison of subspace methods,” IEEE Trans. Power Syst., vol. 28, no. 3, pp. 2907–2917, Aug. 2013. \[24\] J. Zhang and H. Xu, “Microperturbation method for power system online model identification,” IEEE Trans. Ind. Informat., vol. 12, no. 3, pp. 1055–1063, Jun. 2016. \[25\] V. Peric, X. Bombois and L. Vanfretti, “Optimal multisine probing signal’ design for power system electromechanical mode estimation”, In Proc. 50th Hawaii International Conf. on System Sciences, 2017, pp. 1-9. \[26\] S. Boersma, X. Bombois, L. Vanfretti, J. C. Gonzalez-Torres, and A. Benchaib, “Probing signal design for enhanced damping estimation in power networks,” Int. J. Elect. Power Energy Syst., vol. 129, 2021, Art. no. 106640. \[27\] M. de Castro et al., ”Real-Time Prototyping of Optimal Experiment Design in Power Systems using Modelica and FMI,” in Proc. IEEE PES. Gen Meeting, Denver, CO, USA, 2022, pp. 1-5. \[28\] T. L. Vandoorn, F. M. de Belie, T. J. Vyncke, J. A. Melkebeek, and P. Lataire, “Generation of multisinusoidal test signals for the identification of synchronous-machine parameters by using a voltage-source inverter,” IEEE Trans. Ind. Electron., vol. 57, no. 1, pp. 430–439, Jan. 2010. \[29\] J. Zhang and H. Xu, “Online identification of power system equiva lent inertia constant,” IEEE Trans. Ind. Electron., vol. 64, no. 10, pp. 8098–8107, Oct. 2017. \[30\] M. Mukherjee, S. V. Singh, A. M. Kulkarni, K. Dey, and A. M. Gole, “Online estimation of linearized IBR models using ambient noise and external excitation,” Electric Power Syst. Res., vol. 251, 2026, Art. no. 112208. \[31\] Z. Jiang et al., “Probing-based inertia estimation method using hybrid power plants,” in Proc. IEEE Power Energy Soc. Gen. Meeting, Orlando, FL, USA, 2023, pp. 1–5. \[32\] J. Peng et al., ”Probing signal-based inertia and frequency response es timation for power systems with high levels of inverter-based resources,” in Proc. IEEE PES. Gen Meeting, Seattle, WA, USA, 2024, pp. 1-5. \[33\] Z. Shen, M. Jaksic, P. Mattavelli, D. Boroyevich, J. Verhulst, and M. Belkhayat, “Three-phase AC system impedance measurement unit (IMU) using chirp signal injection,” in Proc. 28th Annu. IEEE Appl. Power Electron. Conf. Expo., 2013, pp. 2666–2673. \[34\] P. Pan, H. Hu, D. Xiao, Y. Song, and Z. He, “An improved controlled frequency-band impedance measurement scheme for railway traction power system,” IEEE Trans. Ind. Electron., vol. 68, no. 3, pp. 2184–2195, Mar. 2021. \[35\] H. Cai, L. Yang, H. Wang, P. Song, and Z. Xu, “Application of Unified Power Flow Controller (UPFC) in Jiangsu power system,” in Proc. IEEE PES. Gen Meeting, Chicago, IL, USA, 2017, pp. 1–5. \[36\] W. Wen, K. Li, Z. Zhao, L. Yuan, X. Mo, and W. Cai, “Analysis and control of a four-port megawatt-level high-frequency-bus-based power electronic transformer,” IEEE Trans. Power Electron., vol. 36, no. 11, pp. 13080–13095, Nov. 2021. \[37\] Y. Jiang, Y. Wang, H. Fang, D. Liu and C. Lu, ”A novel prior data selection method for enhancing ambient-signal-based load modeling,” in Proc. IEEE PES. Gen Meeting, Seattle, WA, USA, 2024, pp. 1-5. \[38\] P. Rocca, G. Oliverfi, and A. Massa, “Differential evolution as applied to electromagnetics,” IEEE Antennas Propag. Mag., vol. 53, no. 1, pp. 38–49, Feb. 2011. \[39\] T. Athay, R. Podmore, and S. Virmani, “A practical method for the direct analysis of transient stability,” IEEE Trans. Power App. Syst., vol. PAS-98, no. 2, pp. 573–584, Mar. 1979. \[40\] B. K. Choi and H. D. Chiang, ”Multiple solutions and Plateau phe nomenon in measurement-based load model development: Issues and suggestions,” IEEE Trans. Power Syst., vol. 24, no. 2, pp. 824-831, May 2009.

