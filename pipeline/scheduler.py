import math
from typing import Optional

class CosineCurriculumAnnealingScheduler:
    """
    Curriculum Annealing Scheduler for Privileged Information Training:
    
    1. Alpha (Hint Strength): Decays smoothly from start_alpha (e.g., 1.0) -> end_alpha (e.g., 0.0).
    2. Lambda_C (Conditional Loss Weight): Scales proportionally with alpha: lambda_c_max * alpha.
    3. Lambda_Cons (Consistency Loss Weight): Scales inversely to smoothly increase distillation
       as alpha decays: lambda_cons_max * (1.0 - alpha).
       
    Formula:
        alpha(e) = end_alpha + 0.5 * (start_alpha - end_alpha) * (1 + cos(pi * e / E_total))
        
    Guarantees:
      - When start_alpha=0.0 or hint is disabled, returns alpha=0, lambda_c=0, lambda_cons=0 (100% equivalent to baseline).
      - At epoch >= total_epochs, returns exact end values (0.0).
    """
    def __init__(
        self,
        total_epochs: int = 15,
        start_alpha: float = 1.0,
        end_alpha: float = 0.0,
        lambda_c_max: float = 0.40,
        lambda_cons_max: float = 0.30
    ):
        self.total_epochs = max(1, total_epochs)
        self.start_alpha = float(start_alpha)
        self.end_alpha = float(end_alpha)
        self.lambda_c_max = float(lambda_c_max)
        self.lambda_cons_max = float(lambda_cons_max)

    def step(self, epoch: int) -> dict:
        """
        Returns parameters for current epoch:
          {
            'alpha': float in [end_alpha, start_alpha],
            'lambda_c': float,
            'lambda_cons': float
          }
        """
        if self.start_alpha == 0.0 and self.end_alpha == 0.0:
            return {'alpha': 0.0, 'lambda_c': 0.0, 'lambda_cons': 0.0}

        if epoch >= self.total_epochs:
            alpha = self.end_alpha
        else:
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * epoch / self.total_epochs))
            alpha = self.end_alpha + (self.start_alpha - self.end_alpha) * cos_factor

        lambda_c = self.lambda_c_max * alpha
        lambda_cons = self.lambda_cons_max * (1.0 - alpha)

        return {
            'alpha': float(alpha),
            'lambda_c': float(lambda_c),
            'lambda_cons': float(lambda_cons)
        }
