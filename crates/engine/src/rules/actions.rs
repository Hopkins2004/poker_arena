//! Structure-independent action logic: what a player may do ([`legal_actions`]) and whether
//! the action a bot returned is allowed ([`resolve`]). The betting structure only supplies the
//! raise range and cap through [`BettingRules`].

use super::betting::{BetContext, BettingRules};
use crate::action::{Action, IllegalAction, LegalActions};

/// The acting player's situation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlayerView {
    /// Chips already put in on this street.
    pub street_contrib: u64,
    /// Chips behind.
    pub stack: u64,
    /// Whether the player already acted on this street.
    pub acted: bool,
    /// `current_bet` right after the player last acted on this street.
    pub acted_at: u64,
    /// Whether another player still has chips to respond to a raise.
    pub others_can_respond: bool,
}

/// Computes the legal actions of the player to act.
pub fn legal_actions<B: BettingRules + ?Sized>(
    betting: &B,
    ctx: &BetContext,
    p: &PlayerView,
) -> LegalActions {
    let to_call_raw = ctx.current_bet.saturating_sub(p.street_contrib);
    let call_amount = to_call_raw.min(p.stack);
    let can_check = to_call_raw == 0;
    let allin_to = p.street_contrib + p.stack;

    // Betting is (re)opened for this player if it has not acted on this street, or if it now
    // faces at least a full raise since it last acted (short all-ins don't reopen; TDA
    // cumulative rule).
    let reopened = !p.acted || ctx.current_bet - p.acted_at >= ctx.last_raise;
    let can_raise =
        p.stack > to_call_raw && reopened && p.others_can_respond && betting.raise_permitted(ctx);

    let (mut min_to, mut max_to) = (0, 0);
    if can_raise {
        let (lo, hi) = betting.raise_to_range(ctx);
        // A stack smaller than a full raise may still go all-in (a short all-in).
        min_to = lo.min(allin_to);
        max_to = hi.min(allin_to).max(min_to);
    }
    LegalActions {
        can_check,
        call_amount,
        can_raise,
        min_raise_to: min_to,
        max_raise_to: max_to,
    }
}

/// A validated action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resolved {
    /// Fold.
    Fold,
    /// Check (also a `Call` with nothing to call).
    Check,
    /// Call, possibly all-in for less.
    Call,
    /// Bet or raise to this street total.
    RaiseTo(u64),
}

/// Validates `action` against the player's legal actions. `allin_to` is the player's street
/// contribution plus stack. `BetTo`/`RaiseTo` are interchangeable; `Call` with nothing to call
/// is a check; `AllIn` is a call when it doesn't exceed the current bet, otherwise a raise
/// that must fit the legal range.
pub fn resolve(
    action: Action,
    la: &LegalActions,
    allin_to: u64,
    current_bet: u64,
) -> Result<Resolved, IllegalAction> {
    match action {
        Action::Fold => Ok(Resolved::Fold),
        Action::Check if la.can_check => Ok(Resolved::Check),
        Action::Check => Err(IllegalAction::CannotCheck),
        Action::Call if la.can_check => Ok(Resolved::Check),
        Action::Call => Ok(Resolved::Call),
        Action::BetTo(x) | Action::RaiseTo(x) => {
            if !la.can_raise {
                Err(IllegalAction::RaiseNotAllowed)
            } else if x < la.min_raise_to {
                Err(IllegalAction::BelowMinRaise)
            } else if x > la.max_raise_to {
                Err(IllegalAction::AboveMaxRaise)
            } else {
                Ok(Resolved::RaiseTo(x))
            }
        }
        Action::AllIn if allin_to <= current_bet => Ok(Resolved::Call),
        Action::AllIn if !la.can_raise => Err(IllegalAction::RaiseNotAllowed),
        Action::AllIn if allin_to > la.max_raise_to => Err(IllegalAction::AboveMaxRaise),
        Action::AllIn => Ok(Resolved::RaiseTo(allin_to)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::Street;
    use crate::format::Betting;

    fn ctx(current_bet: u64, last_raise: u64, pot: u64, to_call: u64) -> BetContext {
        BetContext {
            street: Street::Flop,
            bb: 2,
            current_bet,
            last_raise,
            bets_this_street: 1,
            pot,
            to_call,
        }
    }
    fn player(contrib: u64, stack: u64) -> PlayerView {
        PlayerView {
            street_contrib: contrib,
            stack,
            acted: false,
            acted_at: 0,
            others_can_respond: true,
        }
    }

    #[test]
    fn structures_give_their_ranges() {
        // Facing a bet of 10 into 30 (pot incl. the bet = 40), 200 behind.
        let c = ctx(10, 10, 40, 10);
        let p = player(0, 200);
        let nl = legal_actions(&Betting::NoLimit, &c, &p);
        assert_eq!((nl.min_raise_to, nl.max_raise_to), (20, 200));
        let pl = legal_actions(&Betting::PotLimit, &c, &p);
        assert_eq!((pl.min_raise_to, pl.max_raise_to), (20, 60)); // 10 + 40 + 10
        let fl = Betting::FixedLimit {
            small_bet: 4,
            big_bet: 8,
            cap: 4,
        };
        let fla = legal_actions(&fl, &c, &p);
        assert_eq!((fla.min_raise_to, fla.max_raise_to), (14, 14));
        let capped = BetContext {
            bets_this_street: 4,
            ..c
        };
        assert!(!legal_actions(&fl, &capped, &p).can_raise);
    }

    #[test]
    fn short_stack_and_reopening() {
        let c = ctx(10, 10, 40, 10);
        let short = legal_actions(&Betting::NoLimit, &c, &player(0, 15));
        assert_eq!((short.min_raise_to, short.max_raise_to), (15, 15));
        // Acted at 6, now facing 10: increment 4 < full raise 10 -> not reopened.
        let p = PlayerView {
            acted: true,
            acted_at: 6,
            ..player(6, 200)
        };
        assert!(!legal_actions(&Betting::NoLimit, &c, &p).can_raise);
    }

    #[test]
    fn resolve_rules() {
        let la = LegalActions {
            can_check: false,
            call_amount: 10,
            can_raise: true,
            min_raise_to: 20,
            max_raise_to: 60,
        };
        assert_eq!(resolve(Action::Call, &la, 200, 10), Ok(Resolved::Call));
        assert_eq!(
            resolve(Action::Check, &la, 200, 10),
            Err(IllegalAction::CannotCheck)
        );
        assert_eq!(
            resolve(Action::RaiseTo(19), &la, 200, 10),
            Err(IllegalAction::BelowMinRaise)
        );
        assert_eq!(
            resolve(Action::BetTo(60), &la, 200, 10),
            Ok(Resolved::RaiseTo(60))
        );
        assert_eq!(
            resolve(Action::AllIn, &la, 200, 10),
            Err(IllegalAction::AboveMaxRaise)
        );
        assert_eq!(resolve(Action::AllIn, &la, 8, 10), Ok(Resolved::Call));
        let check = LegalActions {
            can_check: true,
            ..LegalActions::default()
        };
        assert_eq!(resolve(Action::Call, &check, 200, 0), Ok(Resolved::Check));
    }
}
