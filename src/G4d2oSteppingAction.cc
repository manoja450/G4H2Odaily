#include "G4d2oSteppingAction.hh"
#include "G4d2oDataDrivenReflector.hh"
#include "inputVariables.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"
#include "G4SystemOfUnits.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4VSolid.hh"
#include "G4AffineTransform.hh"
#include <cmath>

// ============================================================
// Helper: get global surface normal using local coordinates
// ============================================================
static G4ThreeVector GetGlobalSurfaceNormal(const G4StepPoint* point) {
    G4VPhysicalVolume* volume = point->GetPhysicalVolume();
    if (!volume || !volume->GetLogicalVolume() || !volume->GetLogicalVolume()->GetSolid())
        return G4ThreeVector(0, 0, 1);

    G4TouchableHandle touchable = point->GetTouchableHandle();
    G4AffineTransform transform = touchable->GetHistory()->GetTopTransform(); // global -> local

    G4ThreeVector localPos = transform.TransformPoint(point->GetPosition());
    G4ThreeVector localNormal = volume->GetLogicalVolume()->GetSolid()->SurfaceNormal(localPos);

    return transform.Inverse().TransformAxis(localNormal); // rotate normal back to global
}

G4d2oSteppingAction::G4d2oSteppingAction()
    : fReflectionCount(0), fTotalPathLength(0.0), fPhotonCount(0),
      fCurrentPhotonID(-1), fHasPrevDirection(false), fEventID(0) {}

G4d2oSteppingAction::~G4d2oSteppingAction() {}

void G4d2oSteppingAction::ResetEventCounters() {
    fReflectionCount = 0;
    fTotalPathLength = 0.0;
    fPhotonCount = 0;
    fCurrentPhotonID = -1;
    fHasPrevDirection = false;
}

void G4d2oSteppingAction::UserSteppingAction(const G4Step* step) {
    G4Track* track = step->GetTrack();
    G4ParticleDefinition* particle = track->GetDefinition();
    G4String particleName = particle->GetParticleName();

    if (particleName != "opticalphoton") return;

    G4int trackID = track->GetTrackID();
    G4double stepLength = step->GetStepLength();

    if (trackID != fCurrentPhotonID) {
        fCurrentPhotonID = trackID;
        fPhotonCount++;
        fHasPrevDirection = false;
    }

    fTotalPathLength += stepLength;

    G4ThreeVector currentDir = track->GetMomentumDirection();
    if (fHasPrevDirection) {
        G4double dot = fPrevDirection.dot(currentDir);

        // ============================================================
        // FIX: only count reflections if direction changed AND we crossed a real boundary
        // ============================================================
        if (dot < 0.998 && step->GetPostStepPoint()->GetStepStatus() == fGeomBoundary) {
            fReflectionCount++;

            // ============================================================
            // Fill ReflectionTree only for UNIFIED model
            // Data-Driven model fills it inside the custom boundary.
            // ============================================================
            inputVariables* input = inputVariables::GetIVPointer();
            if (input->GetReflectionModel() == 0) { // Unified

                // ---- Optional: check that we're on a Tyvek surface ----
                G4VPhysicalVolume* vol = step->GetPreStepPoint()->GetPhysicalVolume();
                G4VPhysicalVolume* postVol = step->GetPostStepPoint()->GetPhysicalVolume();
                G4String name1 = vol ? vol->GetName() : "";
                G4String name2 = postVol ? postVol->GetName() : "";
                if ( !(name1.contains("tyvek") || name2.contains("tyvek")) ) {
                    // not a Tyvek reflection, skip filling
                    goto skip_fill;
                }

                // ---- Get correct normal using local transform ----
                G4StepPoint* prePoint = step->GetPreStepPoint();
                G4ThreeVector normal = GetGlobalSurfaceNormal(prePoint);
                if (fPrevDirection.dot(normal) > 0) normal = -normal;

                G4double incidentRad = fPrevDirection.angle(-normal);
                G4double incidentDeg = incidentRad / deg;
                // Clamp to [0,90] to match Data‑Driven
                if (incidentDeg > 90.0) incidentDeg = 90.0;
                if (incidentDeg < 0) incidentDeg = 0;

                G4double outDotNormal = currentDir.dot(normal);
                G4double reflectedRad = std::acos(std::min(1.0, std::max(-1.0, outDotNormal)));
                G4double reflectedMag = reflectedRad / deg;

                G4ThreeVector tangent = fPrevDirection - (fPrevDirection.dot(normal)) * normal;
                if (tangent.mag() < 1e-10) {
                    tangent = G4ThreeVector(1, 0, 0);
                    if (std::abs(normal.dot(tangent)) > 0.9999) tangent = G4ThreeVector(0, 1, 0);
                }
                tangent = tangent.unit();
                G4double outTangent = currentDir.dot(tangent);
                G4double sign = (outTangent >= 0) ? 1.0 : -1.0;
                G4double reflectedDeg = sign * reflectedMag;

                // Fill the tree via the reflector's RecordReflection
                G4d2oDataDrivenReflector::RecordReflection(incidentDeg, reflectedDeg,
                                                           currentDir.x(), currentDir.y(), currentDir.z(),
                                                           normal.x(), normal.y(), normal.z());
            }
        }
        skip_fill: ;
    }
    fPrevDirection = currentDir;
    fHasPrevDirection = true;
}
