{
  description = "Nix Development Flake for your package";
  #inputs.nixpkgs.url = "github:NixOS/nixpkgs/master";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    /*poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };*/
  };

  outputs =
    { self, nixpkgs, flake-utils }:

    flake-utils.lib.eachDefaultSystem
      (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python310;
        #pythonPackages = python.pkgs;
      in
      {
        devShells.default = pkgs.mkShell {
          name = "your_package";
          nativeBuildInputs = [ pkgs.bashInteractive ];
          buildInputs = with pkgs; [
            poetry
            #(poetry.override { python = python; })
            (python.withPackages(ps: with ps; [
              setuptools
              wheel
              venvShellHook
            ]))
          ];
          venvDir = ".venv";
          src = null;
          postVenv = ''
            unset SOURCE_DATE_EPOCH
          '';
          postShellHook = ''
            unset SOURCE_DATE_EPOCH
            unset LD_PRELOAD

            PYTHONPATH=$PWD/$venvDir/${python.sitePackages}:$PYTHONPATH
          '';
        };
      });
}
