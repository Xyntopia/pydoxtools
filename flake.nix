{
  description = "Nix Development Flake for Pydoxtools";
  #inputs.nixpkgs.url = "github:NixOS/nixpkgs/master";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
    #nix-ld.url = "github:Mic92/nix-ld";
    # nix-ld.inputs.nixpkgs.follows = "nixpkgs";
    #nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    /*poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };*/
    #https://www.reddit.com/r/NixOS/comments/11xgby8/python_and_flake_infinite_recursion/
  };

  outputs =
    { self, nixpkgs, flake-utils, ... }:

    flake-utils.lib.eachDefaultSystem
      (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python310;
        #pythonPackages = python.pkgs;
      in
      {
        devShells.default = pkgs.mkShell rec {
          name = "pydoxtools_dev_shell";
          nativeBuildInputs = [ pkgs.bashInteractive ];
          buildInputs = with pkgs; [
            poetry
            #(poetry.override { python = python; })
            (python.withPackages(ps: with ps; [
              pip
              setuptools
              wheel
              # venvShellHook
              # for the magic library:
              # python-magic
            ]))
            graphviz
            #libz #for some reason this is too old and conflicts with zlib..  no idea why...
            zlib
            # for magic library if we want to install it using
            # pip install python-libmagic:
            libffi
            #nix-ld.nixosModules.nix-ld
            cmake
            file
          ];
          shellHook = ''
            unset SOURCE_DATE_EPOCH
            unset LD_PRELOAD

            # Environment variables
            # fixes libstdc++ issues, libz.so.1 issues
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib/:${pkgs.lib.makeLibraryPath buildInputs}:$NIX_LD_LIBRARY_PATH";

            echo "setting poetry env for $(which python)"
            poetry env use $(which python)
            #poetry env use 3.11
            #bash -C poetry shell
            echo "Activating poetry environment"
            POETRY_ENV_PATH=$(poetry env list --full-path | grep Activated | cut -d' ' -f1)
            source "$POETRY_ENV_PATH/bin/activate"

            #PYTHONPATH=$PWD/$venvDir/${python.sitePackages}:$PYTHONPATH
          '';
          # fixes xcb issues :
          #QT_PLUGIN_PATH=${qt5.qtbase}/${qt5.qtbase.qtPluginPrefix}

          # fixes libstdc++ issues and libgl.so issues
          #LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib/:/run/opengl-driver/lib/
          #export LD_LIBRARY_PATH="$(nix eval --raw nixpkgs.zlib)/lib${LD_LIBRARY_PATH:+:}${LD_LIBRARY_PATH}"
        };
      });
}
