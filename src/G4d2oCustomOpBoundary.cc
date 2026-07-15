#include "G4d2oCustomOpBoundary.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4OpticalPhoton.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4StepPoint.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4VSolid.hh"
#include "G4ParticleChange.hh"
#include "G4OpticalSurface.hh"
#include <random>
#include <cstdio>

static G4int gPrintCount = 0;
static G4int gMaxPrint = 100;
static G4bool gPrintLimitReached = false;

G4d2oCustomOpBoundary::G4d2oCustomOpBoundary(const G4String& processName)
    : G4OpBoundaryProcess(processName) {
    G4cout << "\n=========================================================" << G4endl;
    G4cout << "G4d2oCustomOpBoundary: Created - PURE DATA-DRIVEN REFLECTOR" << G4endl;
    G4cout << "  Using ONLY thesis data from Chavarria 2007" << G4endl;
    G4cout << "  NO mixing, NO bias, NO artificial fixes" << G4endl;
    G4cout << "=========================================================\n" << G4endl;
}

G4d2oCustomOpBoundary::~G4d2oCustomOpBoundary() {}

G4VParticleChange* G4d2oCustomOpBoundary::PostStepDoIt(const G4Track& track, const G4Step& step) {
    
    const G4StepPoint* postStepPoint = step.GetPostStepPoint();
    G4ThreeVector incomingDirection = track.GetMomentumDirection();
    G4VPhysicalVolume* volume = postStepPoint->GetPhysicalVolume();
    
    G4ThreeVector normal(0, 0, 1);
    if (volume != nullptr && volume->GetLogicalVolume() != nullptr && 
        volume->GetLogicalVolume()->GetSolid() != nullptr) {
        normal = volume->GetLogicalVolume()->GetSolid()->SurfaceNormal(postStepPoint->GetPosition());
    }
    
    G4ThreeVector incoming = incomingDirection.unit();
    G4ThreeVector norm = normal.unit();
    if (incoming.dot(norm) > 0) {
        norm = -norm;
    }
    
    G4double incidentAngleRad = incoming.angle(-norm);
    G4double incidentDeg = incidentAngleRad / deg;
    if (incidentDeg > 90.0) incidentDeg = 90.0;
    if (incidentDeg < 0) incidentDeg = 0;
    
    G4d2oDataDrivenReflector* reflector = G4d2oDataDrivenReflector::GetInstance();
    
    G4ParticleChange* particleChange = new G4ParticleChange();
    particleChange->Initialize(track);
    
    if (reflector != nullptr) {
        // ============================================================
        // PURE DATA-DRIVEN: Sample directly from thesis PDFs
        // NO mixing with Lambertian
        // NO upward bias
        // NO artificial fixes
        // ============================================================
        G4double thetaOutDeg = reflector->SampleOutgoingAngle(incidentDeg);
        G4double thetaOutRad = thetaOutDeg * deg;
        
        // Record for ROOT
        reflector->RecordReflection(incidentDeg, thetaOutDeg);
        
        // Limited console printing
        if (!gPrintLimitReached) {
            if (gPrintCount < gMaxPrint) {
                gPrintCount++;
                printf("  [PURE DATA %d] incident=%.2f°, reflected=%.2f°\n", 
                       gPrintCount, incidentDeg, thetaOutDeg);
                fflush(stdout);
            } else if (gPrintCount == gMaxPrint) {
                gPrintCount++;
                printf("  ... (further reflections not printed)\n");
                fflush(stdout);
            }
        }
        
        // Calculate reflection direction in plane of incidence
        G4ThreeVector perp = incoming - (incoming.dot(norm)) * norm;
        if (perp.mag() < 1e-10) {
            perp = G4ThreeVector(1, 0, 0);
            if (std::abs(norm.dot(perp)) > 0.9999) perp = G4ThreeVector(0, 1, 0);
        }
        perp = perp.unit();
        
        // Construct reflected direction from thesis-sampled angle
        G4ThreeVector reflected = -std::cos(thetaOutRad) * norm + std::sin(thetaOutRad) * perp;
        
        // Random azimuth (physically correct - isotropic out-of-plane)
        std::uniform_real_distribution<G4double> dist(0.0, 2.0 * M_PI);
        static std::mt19937 rng(std::random_device{}());
        G4double azimuth = dist(rng);
        
        G4ThreeVector perp2 = norm.cross(perp).unit();
        G4ThreeVector finalDir = std::cos(azimuth) * reflected + std::sin(azimuth) * perp2;
        finalDir = finalDir.unit();
        
        particleChange->ProposeMomentumDirection(finalDir);
        particleChange->ProposeEnergy(track.GetKineticEnergy());
        particleChange->ProposeLocalEnergyDeposit(0.0);
        particleChange->SetNumberOfSecondaries(0);
        
        return particleChange;
    } else {
        return G4OpBoundaryProcess::PostStepDoIt(track, step);
    }
}
