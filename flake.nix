{
  description = "10-708 course project";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      inherit (nixpkgs) lib;
      systems = lib.systems.flakeExposed;
      eachDefaultSystem = f: builtins.foldl' lib.attrsets.recursiveUpdate { }
        (map f systems);
    in
    eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python' = pkgs.python3.withPackages (ps: with ps; [
          fenics
          jax
          jaxlib
          jupyter
          pycairo
          seaborn
        ]);
        formatters = [ pkgs.black pkgs.isort pkgs.nixpkgs-fmt ];
        linters = [ pkgs.pyright pkgs.ruff pkgs.statix ];
        # pkgs/build-support/build-fhsenv-bubblewrap/buildFHSEnv.nix
        etcFishConfig = pkgs.writeText "config.fish" ''
          # >>> mamba initialize >>>
          # !! Contents within this block are managed by 'mamba init' !!
          set -gx MAMBA_EXE "micromamba"
          set -gx MAMBA_ROOT_PREFIX "$HOME/micromamba"
          $MAMBA_EXE shell hook --shell fish \
            --root-prefix $MAMBA_ROOT_PREFIX | source
          # <<< mamba initialize <<<
        '';
        etcFishPkg = pkgs.runCommandLocal "fish-chrootenv-etc" { } ''
          mkdir -p $out/etc/fish
          pushd $out/etc/fish

          ln -s ${etcFishConfig} config.fish
        '';
        # https://nixos.wiki/wiki/Python#micromamba
        # https://discourse.nixos.org/t/nix-shell-with-micromamba-and-fhs/25464
        fhs = pkgs.buildFHSEnv {
          name = "micromamba";

          targetPkgs = pkgs: (with pkgs; [
            etcFishPkg
            fish
            micromamba
          ]);

          profile = ''
            set -e
            # prevent leaking nix dependencies
            export PYTHONPATH=
            # make sure prompt fits on the line
            export MAMBA_ENV_PROMPT='"({name}) "'
            eval "$(micromamba shell hook --shell posix)"
            venv='./.venv'
            if test -d "$venv"; then
              micromamba activate "$venv"
            fi
            set +e
          '';

          runScript = "fish";
        };
        fhs-shell = "${fhs}/bin/${fhs.name}";
      in
      {
        formatter.${system} = pkgs.writeShellApplication {
          name = "formatter";
          runtimeInputs = formatters;
          text = ''
            isort "$@"
            black "$@"
            nixpkgs-fmt "$@"
          '';
        };

        checks.${system}.lint = pkgs.stdenvNoCC.mkDerivation {
          name = "lint";
          src = ./.;
          doCheck = true;
          nativeCheckInputs = formatters ++ linters ++ lib.singleton python';
          checkPhase = ''
            isort --check --diff .
            black --check --diff .
            nixpkgs-fmt --check .
            ruff check .
            pyright .
            statix check
          '';
          installPhase = "touch $out";
        };

        apps.${system} = {
          default = {
            type = "app";
            program = fhs-shell;
          };
        };

        devShells.${system}.default = (pkgs.mkShell.override {
          stdenv = pkgs.clangStdenv;
        }) {
          packages = [
            python'
          ]
          ++ formatters
          ++ linters;
        };
      }
    );
}
