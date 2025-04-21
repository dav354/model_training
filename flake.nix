{
  description = "Dev shell for the custom model";

  outputs = {
    self,
    nixpkgs,
    ...
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
      };
    };
    pythonEnv = pkgs.python310.withPackages (ps: with ps; [    
        opencv-python
        numpy
        tensorflow
        mediapipe
        scikit-learn
      ]);
  in {
    devShells.${system}.default = pkgs.mkShell {
      buildInputs = [
        pythonEnv  
        pkgs.edgetpu-compiler
      ];
    };
  };
}
