/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright held by original author
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software; you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation; either version 2 of the License, or (at your
    option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM; if not, write to the Free Software Foundation,
    Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA

\*---------------------------------------------------------------------------*/

#include "fileBasedBodyForce.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defineTypeNameAndDebug(fileBasedBodyForce, 0);
addToRunTimeSelectionTable(bodyForcing, fileBasedBodyForce, bodyForcing);

// * * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * //

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

fileBasedBodyForce::fileBasedBodyForce
(
    const fvMesh & mesh,
    const dictionary & dict
)
:
    bodyForcing(mesh, dict),
    offset_( readScalar( bodyForcingDict_.lookup("constantSlope") ) ),
    direction_( vector( bodyForcingDict_.lookup("forceDirection" ) ))

{
    direction_ /= Foam::mag(direction_);
}


fileBasedBodyForce::~fileBasedBodyForce()
{}

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

dimensionedVector fileBasedBodyForce::correct( )
{   
    scalar derivative_u;
    Info << "Time = " << mesh_.time().value() << nl << endl;
        // Construct the file path for the current timestep
        // Calculate the effective time by wrapping around every 7 seconds
    double effectiveTime = fmod(mesh_.time().value(), 7.0);
    effectiveTime = round(effectiveTime * 100000000.0) / 100000000;

    // Define a small threshold, e.g., 1e-9
    const double threshold = 1e-3;

    if (effectiveTime < threshold) {
        effectiveTime = 0.0;
    }

        // Determine the directory name: use "0" if the time is an exact multiple of 7, otherwise use the effective time
    word timeKey = (effectiveTime == 0.0) ? "0" : Foam::name(effectiveTime);
    
    if (timeKey.size() >= 5) {
        timeKey = timeKey.substr(0, 5); // Start at index 0, take 3 characters
    } else {
        // If less than 3 characters, keep the whole string as is (this else block is actually redundant)
        timeKey = timeKey; // No change needed, can optionally remove this else block
    }
        try
        {
        IOdictionary simpleDict
        (
            IOobject
            (
                "simpleDict",
                "constant",
                mesh_,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            )
        );
        derivative_u = (readScalar(simpleDict.lookup(timeKey)));
        Info << "Body Force at timestep " << mesh_.time().value() << ": " << derivative_u << endl;
        }
        catch (const Foam::error& e)
        {
            Info << "Unable to read BodyForce for timestep " << mesh_.time().value() << ". Error: " << e.message() << endl;
        }
        Info << "Offset: " << offset_ << nl << endl;
        Info << "Direction: " << direction_ << nl << endl;
        Info << "Derivative_u: " << derivative_u << nl << endl;

    dimensionedVector forcing
    (
        "forcing",
        dimensionSet(0,1,-2,0,0,0,0),
        (
            derivative_u
            + offset_
        ) * direction_
    );

    return forcing;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
