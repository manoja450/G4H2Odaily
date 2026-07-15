#ifndef G4d2oCustomOpBoundary_h
#define G4d2oCustomOpBoundary_h

#include "G4OpBoundaryProcess.hh"
#include "G4d2oDataDrivenReflector.hh"

class G4d2oCustomOpBoundary : public G4OpBoundaryProcess {
public:
    G4d2oCustomOpBoundary(const G4String& processName = "G4d2oCustomOpBoundary");
    virtual ~G4d2oCustomOpBoundary();
    
    virtual G4VParticleChange* PostStepDoIt(const G4Track& track, const G4Step& step) override;
};

#endif
