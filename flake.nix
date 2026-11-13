{
  description = "Audio analyzer dev shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs = { self, nixpkgs }:
    let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
    in {
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = with pkgs; [
          python312
          uv
          ffmpeg
          libsndfile
          # Rust toolchain: deepfilterlib has no cp312 wheel and must
          # be built from source.
          rustc
          cargo
        ];

        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.libsndfile
          pkgs.zlib
          # NVIDIA driver's libcuda, needed by pip-installed CUDA torch
          # (not in ldconfig on NixOS). makeLibraryPath adds "/lib".
          "/run/opengl-driver"
        ];

        shellHook = ''
          if [ ! -d .venv ]; then uv venv; fi
          source .venv/bin/activate
        '';
      };
    };
}
