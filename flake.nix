{
  description = "A Nix-flake-based Multimedia development environment";

  inputs.nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1.*.tar.gz";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f rec {
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python312.withPackages (pp: [
          pp.ipython
          pp.matplotlib
          pp.numba
          pp.numpy
          pp.scipy
        ]);
        pythonPackages = pkgs.python312Packages;
      });
    in
    {
      devShells = forEachSupportedSystem ({ pkgs, python, pythonPackages }: {
        default = pkgs.mkShell {
          venvDir = ".venv";
          packages = [
            python
            pythonPackages.pip
            pythonPackages.venvShellHook

            pkgs.entr
            pkgs.ffmpeg
            pkgs.gnumake
            pkgs.imv
            pkgs.pandoc
            pkgs.pandoc-include
            pkgs.parallel
            pkgs.pigz
            pkgs.qiv
            pkgs.texliveFull
            pkgs.unzip
            pkgs.zip
          ] ++ [
            pkgs.tinymist
            pkgs.typst
            pkgs.typstfmt
            pkgs.typst-live
            pkgs.typstwriter
            pkgs.typstyle
          ];

          LD_LIBRARY_PATH = "${pkgs.portaudio}/lib";
        };

        pedro = pkgs.mkShell {
          venvDir = ".venv";
          packages = [
            python
            pythonPackages.pip
            pythonPackages.venvShellHook

            pkgs.ffmpeg
            pkgs.gnumake
            pkgs.pandoc
            pkgs.pandoc-include
            pkgs.pigz
            pkgs.texliveFull
          ] ++ [
            pkgs.tinymist
            pkgs.typst
            pkgs.typstfmt
            pkgs.typst-live
            pkgs.typstwriter
            pkgs.typstyle
          ];
        };

        LD_LIBRARY_PATH = "${pkgs.portaudio}/lib";
      });
    };
}
