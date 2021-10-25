auxiliary_channels = [
    "SINBAD.RF/LLRF.CONTROLLER/VS.AR.LI.RSB.G.1/AMPL.SAMPLE",       # Gun amplitude
    "SINBAD.RF/LLRF.CONTROLLER/VS.AR.LI.RSB.G.1/PHASE.SAMPLE",      # Gun phase
    "SINBAD.RF/LLRF.CONTROLLER/PROBE.AR.LI.RSB.G.1/POWER.SAMPLE",   # Gun power
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMSOG1+-/FIELD.RBV",               # Solenoid field at center
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMSOG1+-/CURRENT.RBV",             # Solenoid current
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMSOG1+-/MOMENTUM.SP",             # Solenoid beam momentum setting
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG1/KICK.RBV",                  # HCor (ARLIMCHG1) kick
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG1/CURRENT.RBV",               # HCor (ARLIMCHG1) current
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG1/MOMENTUM.SP",               # HCor (ARLIMCHG1) beam momentum setting
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG1/KICK.RBV",                  # VCor (ARLIMCVG1) kick
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG1/CURRENT.RBV",               # VCor (ARLIMCVG1) current
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG1/MOMENTUM.SP",               # VCor (ARLIMCVG1) beam momentum setting
    "SINBAD.UTIL/INJ.LASER.MOTOR/MOTOR1.MBDEV1/POS",                # Laser attenuation
    "SINBAD.LASER/SINBADCPULASER1.SETTINGS/SINBAD_aperture_pos/SETTINGS.CURRENT",   # Laser aperture
    "SINBAD.DIAG/SCREEN.MOTOR/MOTOR4.MBDEV4/POS",                   # Collimator x position
    "SINBAD.DIAG/COLLIMATOR.ML/AR.LI.SLH.G.1/POS",                  # Collimator y position
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG2/KICK.RBV",                  # HCor (ARLIMCHG2) kick
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG2/CURRENT.RBV",               # HCor (ARLIMCHG2) current
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG2/MOMENTUM.SP",               # HCor (ARLIMCHG2) beam momentum setting
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG2/KICK.RBV",                  # VCor (ARLIMCVG2) kick
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG2/CURRENT.RBV",               # VCor (ARLIMCVG2) current
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG2/MOMENTUM.SP",               # VCor (ARLIMCVG2) beam momentum setting
    "SINBAD.DIAG/DARKC_MON/AR.LI.BCM.G.1/CHARGE.CALC",              # Dark current charge (?) -> Actually charge
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMSOL1/CURRENT.RBV",             # TWS1 current
    "SINBAD.RF/LLRF.CONTROLLER/CTRL.AR.LI.RSB.L.1/SP.PHASE",        # TWS1 phase
    "SINBAD.RF/LLRF.CONTROLLER/FORWARD.AR.LI.RSB.L.1/POWER.SAMPLE", # TWS1 power
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG3/KICK.RBV",                  # HCor (ARLIMCHG3) kick
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG3/CURRENT.RBV",               # HCor (ARLIMCHG3) current
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG3/MOMENTUM.SP",               # HCor (ARLIMCHG3) beam momentum setting
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG3/KICK.RBV",                  # VCor (ARLIMCVG3) kick
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG3/CURRENT.RBV",               # VCor (ARLIMCVG3) current
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG3/MOMENTUM.SP",               # VCor (ARLIMCVG3) beam momentum setting
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHM1/KICK.RBV",                  # HCor (ARLIMCHM1) kick
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHM1/CURRENT.RBV",               # HCor (ARLIMCHM1) current
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHM1/MOMENTUM.SP",               # HCor (ARLIMCHM1) beam momentum setting
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVM1/KICK.RBV",                  # VCor (ARLIMCVM1) kick
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVM1/CURRENT.RBV",               # VCor (ARLIMCVM1) current
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVM1/MOMENTUM.SP",               # VCor (ARLIMCVM1) beam momentum setting
    "SINBAD.RF/LLRF.CONTROLLER/CTRL.AR.LI.RSB.L.2/SP.AMPL",         # TWS2 current
    "SINBAD.RF/LLRF.CONTROLLER/CTRL.AR.LI.RSB.L.2/SP.PHASE",        # TWS2 phase
    "SINBAD.RF/LLRF.CONTROLLER/FORWARD.AR.LI.RSB.L.2/POWER.SAMPLE", # TWS2 power
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG4/KICK.RBV",                  # HCor (ARLIMCHG4) kick
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG4/CURRENT.RBV",               # HCor (ARLIMCHG4) current
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCHG4/MOMENTUM.SP",               # HCor (ARLIMCHG4) beam momentum setting
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG4/KICK.RBV",                  # VCor (ARLIMCVG4) kick
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG4/CURRENT.RBV",               # VCor (ARLIMCVG4) current
    "SINBAD.MAGNETS/MAGNET.ML/ARLIMCVG4/MOMENTUM.SP",               # VCor (ARLIMCVG4) beam momentum setting
    "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/STRENGTH.RBV",              # Q1 K1
    "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/CURRENT.RBV",               # Q1 current
    "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM1/MOMENTUM.SP",               # Q1 beam momentum setting
    "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/STRENGTH.RBV",              # Q2 K1
    "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/CURRENT.RBV",               # Q2 current
    "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM2/MOMENTUM.SP",               # Q2 beam momentum setting
    "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/KICK_MRAD.RBV",             # CV kick (mrad)
    "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/CURRENT.RBV",               # CV current
    "SINBAD.MAGNETS/MAGNET.ML/AREAMCVM1/MOMENTUM.SP",               # CV beam momentum setting
    "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/STRENGTH.RBV",              # Q3 K1
    "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/CURRENT.RBV",               # Q3 current
    "SINBAD.MAGNETS/MAGNET.ML/AREAMQZM3/MOMENTUM.SP",               # Q3 beam momentum setting
    "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/KICK_MRAD.RBV",             # CH kick (mrad)
    "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/CURRENT.RBV",               # CH current
    "SINBAD.MAGNETS/MAGNET.ML/AREAMCHM1/MOMENTUM.SP",               # CH beam momentum setting
    "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGHORIZONTAL",           # Screen horizontal binning 
    "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/BINNINGVERTICAL",             # Screen vertical binning
    "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/WIDTH",                       # Screen image width (pixel)
    "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/HEIGHT",                      # Screen image height (pixel)
    "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/GAINRAW",                     # Screen camera gain
    "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/GAINAUTO",                    # Screen camera auto gain setting
    "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/IMAGE_EXT_ZMQ",               # Screen image
    "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/SPECTRUM.X.MEAN",             # Beam mu x
    "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/SPECTRUM.X.MEAN",             # Beam mu y
    "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/SPECTRUM.X.SIG",              # Beam sigma x
    "SINBAD.DIAG/CAMERA/AR.EA.BSC.R.1/SPECTRUM.Y.SIG",              # Beam sigma y
]