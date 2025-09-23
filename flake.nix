{
  description = "OTP-2 thrust analysis development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          pandas
          matplotlib
          numpy
          # Additional useful packages for data analysis
          jupyter
          ipython
          seaborn
          scipy
          pykalman
          # Testing
          pytest
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            pythonEnv
            # Development tools
            git
            gnumake
            # For GUI applications (matplotlib plots)
            pkg-config
            cairo
            pango
            gdk-pixbuf
            gtk3
            gobject-introspection
          ];

          shellHook = ''
            echo "OTP-2 thrust analysis development environment"
            echo "Python: $(python --version)"
            echo "Available packages: pandas, matplotlib, numpy, jupyter, seaborn, scipy"
            echo ""
            echo "Usage:"
            echo "  make plot          - Run the satellite data plotting script"
            echo "  make venv-plot     - Run in virtual environment"
            echo "  make help          - Show all available targets"
            echo ""

            # Set up environment for GUI applications
            export GDK_PIXBUF_MODULE_FILE="${pkgs.librsvg.out}/lib/gdk-pixbuf-2.0/2.10.0/loaders.cache"
          '';

          # Environment variables for Python GUI applications
          QT_QPA_PLATFORM_PLUGIN_PATH = "${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc
            pkgs.cairo
            pkgs.pango
            pkgs.gdk-pixbuf
            pkgs.gtk3
          ];
        };
      });
}
