/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019-2020 OpenCFD Ltd.
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

\*---------------------------------------------------------------------------*/

#include "frozenPropTime_stable.H"
#include "fvOptions.H"
#include "bound.H"

namespace Foam
{
namespace RASModels
{

template<class BasicTurbulenceModel>
void frozenPropTime_stable<BasicTurbulenceModel>::correctNut()
{
    this->nut_ = k_/omega_;
    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);

    BasicTurbulenceModel::correctNut();
}

template<class BasicTurbulenceModel>
frozenPropTime_stable<BasicTurbulenceModel>::frozenPropTime_stable
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    eddyViscosity<RASModel<BasicTurbulenceModel>>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),

    Cmu_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "betaStar",
            this->coeffDict_,
            0.09
        )
    ),
    beta_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "beta",
            this->coeffDict_,
            0.072
        )
    ),
    gamma_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "gamma",
            this->coeffDict_,
            0.52
        )
    ),
    alphaK_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alphaK",
            this->coeffDict_,
            0.5
        )
    ),
    alphaOmega_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alphaOmega",
            this->coeffDict_,
            0.5
        )
    ),

    k_
    (
        IOobject
        (
            IOobject::groupName("k", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    omega_
    (
        IOobject
        (
            IOobject::groupName("omega", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),

    bijDelta_
    (
        IOobject
        (
            "bijDelta",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),

    kDeficit_
    (
        IOobject
        (
            "kDeficit",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    
    aijBoussinesq_
    (
        IOobject
        (
            "aijBoussinesq",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        0.0*symm(fvc::grad(this->U_))*this->nut_
    ),
    aijDelta_
    (
        IOobject
        (
            "aijDelta",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        0.0*symm(fvc::grad(this->U_))*this->nut_
    ),
    Pk_
    (
        IOobject
        (
            "Pk",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "Pk",
        dimensionSet(0,2,-3,0,0,0,0),
        0.0
        )
    ),
    PkDelta_
    (
        IOobject
        (
            "PkDelta",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "PkDelta",
        dimensionSet(0,2,-3,0,0,0,0),
        0.0
        )
    ),
    turbulentProductionTerm
    (
        IOobject
        (
            "turbulentProductionTerm",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "turbulentProductionTerm",
        dimensionSet(0,2,-3,0,0,0,0),
        0.0
        )
    ),
    originalProductionTerm
    (
        IOobject
        (
            "originalProductionTerm",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar
        (
        "originalProductionTerm",
        dimensionSet(0,2,-3,0,0,0,0),
        0.0
        )
    )
{
    IOdictionary dict
    (
        IOobject
        (
            "frozenDict",
            this->runTime_.system(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
    scalar usek = dict.subDict("parameters").lookupOrDefault<scalar>("usek", 1.0);

    Info << "usek: " << usek << nl;

    bound(k_, this->kMin_);
    bound(omega_, this->omegaMin_);

    if (type == typeName)
    {
        this->printCoeffs(type);
    }
}

template<class BasicTurbulenceModel>
Foam::tmp<Foam::fvVectorMatrix>
frozenPropTime_stable<BasicTurbulenceModel>::divDevRhoReff
(
    volVectorField& U
) const
{
    return
    (
        - fvc::div((this->alpha_*this->rho_*this->nuEff())*dev2(T(fvc::grad(U))))
        - fvm::laplacian(this->alpha_*this->rho_*this->nuEff(), U)
        + fvc::div(this->alpha_*this->rho_*dev(2.0*this->k_*bijDelta_))
    );
}

template<class BasicTurbulenceModel>
Foam::tmp<Foam::fvVectorMatrix>
frozenPropTime_stable<BasicTurbulenceModel>::divDevReff
(
    volVectorField& U
) const
{
    return
    (
        - fvc::div((this->alpha_*this->nuEff())*dev2(T(fvc::grad(U))))
        - fvm::laplacian(this->alpha_*this->nuEff(), U)
        + fvc::div(this->alpha_*dev(2.0*this->k_*bijDelta_))
    );
}

template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
frozenPropTime_stable<BasicTurbulenceModel>::devReff() const
{
    return tmp<volSymmTensorField>
    (
        new volSymmTensorField
        (
            IOobject
            (
                IOobject::groupName("devRhoReff", this->alphaRhoPhi_.group()),
                this->runTime_.timeName(),
                this->mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            (-(this->alpha_*this->rho_*this->nuEff()))
           *dev(twoSymm(fvc::grad(this->U_)))
            + dev(2.0*this->k_*this->bijDelta_)
        )
    );
}

template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
frozenPropTime_stable<BasicTurbulenceModel>::R() const
{
    Info << "Custom R() function called." << endl;
    return tmp<volSymmTensorField>
    (
        new volSymmTensorField
        (
            IOobject
            (
                "Rtbnn",
                this->runTime_.timeName(),
                this->mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ), 
            ((2.0/3.0)*I)*this->k_ - this->nut_*twoSymm(fvc::grad(this->U_)) + 2.0*this->k_*this->bijDelta_,
            this->k_.boundaryField().types()
        )
    );
}

template<class BasicTurbulenceModel>
bool frozenPropTime_stable<BasicTurbulenceModel>::read()
{
    if (eddyViscosity<RASModel<BasicTurbulenceModel>>::read())
    {
        Cmu_.readIfPresent(this->coeffDict());
        beta_.readIfPresent(this->coeffDict());
        gamma_.readIfPresent(this->coeffDict());
        alphaK_.readIfPresent(this->coeffDict());
        alphaOmega_.readIfPresent(this->coeffDict());

        return true;
    }

    return false;
}

template<class BasicTurbulenceModel>
void frozenPropTime_stable<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    double effectiveTime = fmod(this->runTime_.value(), 7.0);

    const double threshold = 1e-3;

    if (effectiveTime < threshold) {
        effectiveTime = 0.0;
    }

    word directoryName = (effectiveTime == 0.0) ? "0" : Foam::name(effectiveTime);
    
    word RFieldPath = "reynoldStress/" + directoryName;
    Info << "Using RFieldPath: " << RFieldPath << endl;

    volSymmTensorField bijDelta2_
    (
        IOobject
        (
            "bijDelta",
            RFieldPath,
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_
    );
    volScalarField kDeficit2_
    (
        IOobject
        (
            "kDeficit",
            RFieldPath,
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_
    );
    bijDelta_ = bijDelta2_;
    kDeficit_ = kDeficit2_;

    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
    const volVectorField& U = this->U_;
    const volScalarField& nut = this->nut_;

    fv::options& fvOptions(fv::options::New(this->mesh_));
    
    eddyViscosity<RASModel<BasicTurbulenceModel>>::correct();

    const volScalarField::Internal divU
    (
        fvc::div(fvc::absolute(this->phi(), U))().v()
    );
    
    tmp<volTensorField> tgradU = fvc::grad(U);
    const volScalarField S2(2*magSqr(symm(tgradU())));

    volScalarField GbyNu (dev(twoSymm(tgradU())) && tgradU());
    volScalarField::Internal G(this->GName(), nut()* GbyNu) ;

    PkDelta_ = 2.0*this->k_*(bijDelta_) && symm(tgradU());
    Pk_ = nut*S2 - PkDelta_;

    volScalarField G2(this->GName(), Pk_);
   
    tgradU.clear();

    omega_.boundaryFieldRef().updateCoeffs();
    dimensionedScalar nutSmall
    (
        "nutSmall",
        dimensionSet(0, 2, -1, 0, 0, 0 ,0),
        1e-10
    );

    scalar heightThreshold = 0.01;

    volScalarField mask
    (
        IOobject
        (
            "mask",
            this->mesh_.time().timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("zero", dimless, 0.0)
    );

    forAll(mask, i)
    {
        const scalar y = this->mesh_.C()[i].y();
        if (y < heightThreshold)
        {
            mask[i] = 1.0;
        }
    }

    volScalarField maskedKDeficit = mask * kDeficit_;

    turbulentProductionTerm = alpha()*rho()*G2
                                            + alpha()*rho()*kDeficit_;
    
    originalProductionTerm = alpha()*rho()*G2;

    forAll(turbulentProductionTerm, i)
    {
        if (turbulentProductionTerm[i] < -0.0001)
        {
            turbulentProductionTerm[i] = originalProductionTerm[i];
        }
    }

    tmp<fvScalarMatrix> omegaEqn
    (
        fvm::ddt(alpha, rho, omega_)
      + fvm::div(alphaRhoPhi, omega_)
      - fvm::laplacian(alpha*rho*DomegaEff(), omega_)
     ==
        gamma_*alpha()*rho()*G2/(nut +nutSmall)
      + gamma_*alpha()*rho()*kDeficit_/(nut +nutSmall)
      - fvm::SuSp(((2.0/3.0)*gamma_)*alpha()*rho()*divU, omega_)
      - fvm::Sp(beta_*alpha()*rho()*omega_(), omega_)
      + fvOptions(alpha, rho, omega_)
    );

    omegaEqn.ref().relax();
    fvOptions.constrain(omegaEqn.ref());
    omegaEqn.ref().boundaryManipulate(omega_.boundaryFieldRef());
    solve(omegaEqn);
    fvOptions.correct(omega_);
    bound(omega_, this->omegaMin_);

    tmp<fvScalarMatrix> kEqn
    (
        fvm::ddt(alpha, rho, k_)
      + fvm::div(alphaRhoPhi, k_)
      - fvm::laplacian(alpha*rho*DkEff(), k_)
     ==
        alpha()*rho()*G2
      + alpha()*rho()*kDeficit_
      - fvm::SuSp((2.0/3.0)*alpha()*rho()*divU, k_)
      - fvm::Sp(Cmu_*alpha()*rho()*omega_(), k_)
      + fvOptions(alpha, rho, k_)
    );

    kEqn.ref().relax();
    fvOptions.constrain(kEqn.ref());
    solve(kEqn);
    fvOptions.correct(k_);
    forAll(k_.internalField(), i)
    {
        if (k_[i] < 0.0)
        {
            Info << "Warning: k is negative at cell " << i << " with value " << k_[i] << endl;
        }
    }

    bound(k_, this->kMin_);

    correctNut();
    aijBoussinesq_ = -nut*twoSymm(fvc::grad(U));
    aijDelta_ = -2.0*k_*bijDelta_;
}

} 
} 
