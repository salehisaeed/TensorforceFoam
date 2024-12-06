/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::agentJetFvPatchVectorField

See also
    Foam::fixedValueFvPatchField
    Foam::Function1Types

SourceFiles
    agentJetFvPatchVectorField.C

\*---------------------------------------------------------------------------*/

#ifndef agentJetFvPatchVectorField_H
#define agentJetFvPatchVectorField_H

#include <cppflow/ops.h>
#include <cppflow/model.h>
#include "fixedValueFvPatchFields.H"
#include "writeFile.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
           Class agentJetFvPatchVectorField Declaration
\*---------------------------------------------------------------------------*/

class agentJetFvPatchVectorField
:
    public fixedValueFvPatchVectorField,
    public functionObjects::writeFile
{
    // Private data

        //- deterministic or stochastic policy
        bool deterministic_;

        //- Controller time step
        scalar controlPeriod_;

        //- Times to ramp-up a new controller value must be less than or equal to controlInterval_
        scalar rampUpPeriod_;

        //- Next action
        scalar actionNew_;

        //- Previously predicted angular velocity
        scalar actionOld_;

        //- Current time index (used for updating)
        label curTimeIndex_;

        //- State field name
        word stateFieldName_;

        //- Number of the state probes
        label stateProbesNo_;

        //- Locations of the state probes
        vectorField stateProbeLocations_;

        //- Interpolation scheme of the probes
        word interpolationScheme_;

        //- Directory name of the saved TensorFlow policy model
        fileName policyDirName_;

        //- DRL model
        cppflow::model model_;


    // Private Member Functions

        //- Get environment state
        scalarField environmentState();

        //- Make agent action
        scalar agentAction(const scalarField& state);

        //- Save trajectory to file
        void writeStateAction(const scalarField& state, const scalar actionNew);

        //- Output file header information
        virtual void writeFileHeader(Ostream& os);


    public:
        //- Runtime type information
        TypeName("agentJet");

        // Constructors

            //- Construct from patch and internal field
            agentJetFvPatchVectorField
            (
                const fvPatch&,
                const DimensionedField<vector, volMesh>&
            );

            //- Construct from patch, internal field and dictionary
            agentJetFvPatchVectorField
            (
                const fvPatch&,
                const DimensionedField<vector, volMesh>&,
                const dictionary&
            );

            //- Construct by mapping given agentJetFvPatchVectorField
            //  onto a new patch
            agentJetFvPatchVectorField
            (
                const agentJetFvPatchVectorField&,
                const fvPatch&,
                const DimensionedField<vector, volMesh>&,
                const fvPatchFieldMapper&
            );

            //- Construct as copy
            agentJetFvPatchVectorField
            (
                const agentJetFvPatchVectorField&
            );

            //- Construct and return a clone
            virtual tmp<fvPatchVectorField> clone() const
            {
                return tmp<fvPatchVectorField>
                (
                    new agentJetFvPatchVectorField(*this)
                );
            }

            //- Construct as copy setting internal field reference
            agentJetFvPatchVectorField
            (
                const agentJetFvPatchVectorField&,
                const DimensionedField<vector, volMesh>&
            );

            //- Construct and return a clone setting internal field reference
            virtual tmp<fvPatchVectorField> clone
            (
                const DimensionedField<vector, volMesh>& iF
            ) const
            {
                return tmp<fvPatchVectorField>
                (
                    new agentJetFvPatchVectorField(*this, iF)
                );
            }


    // Member functions

        //- Update the coefficients associated with the patch field
        virtual void updateCoeffs();

        //- Write
        virtual void write(Ostream&) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
