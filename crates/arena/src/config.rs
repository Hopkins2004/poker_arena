//! TOML run configuration.
//!
//! ```toml
//! [run]
//! rounds = 5
//! seed = 42
//! # threads = 8                    # default: all cores
//!
//! [format]
//! variant = "nlhe"                  # a registered format: nlhe | plo | flhe
//! max_seats = 6
//! # small_bet = 2, big_bet = 4, cap = 4   (flhe only)
//!
//! [table]
//! hands_per_table = 1000
//! starting_stack_bb = 100
//! mode = "cash"                     # cash | tournament
//! blinds = { sb = 1, bb = 2, ante = 0 }   # cash
//! # levels = [{ sb = 1, bb = 2, ante = 0, hands = 50 }, ...]   # tournament
//! stack_policy = "reset_each_hand"  # reset_each_hand | rebuy_when_busted | top_up_to_start
//! duplicate_mode = false
//! illegal_action_policy = "fold_or_check"  # fold_or_check | error
//! record_timing = false
//! hand_history = false
//!
//! [matchmaking]
//! min_appearances = 4
//! allow_duplicate_bot_at_table = true
//!
//! [rating]
//! k = 16
//! score_mode = "binary"             # binary | margin
//!
//! [[bots]]
//! name = "tag"
//! version = "1"                     # optional
//! kind = "python:pokerarena.baselines.TightAggressive"   # python:module.ClassName
//! anchor = 1300                     # optional fixed Elo
//! params = { samples = 300 }
//! ```

use crate::matchmaking::MatchmakingConfig;
use crate::rating::RatingConfig;
use crate::runner::RunSpec;
use crate::table::{BlindLevel, Blinds, IllegalActionPolicy, StackPolicy, TableConfig, TableMode};
use engine::{Betting, GameFormat};
use serde::{Deserialize, Serialize};

/// Run-level settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RunSection {
    /// Number of rounds.
    pub rounds: usize,
    /// Master seed.
    pub seed: u64,
    /// Worker threads (`None` = all cores).
    pub threads: Option<usize>,
    /// Directory for per-run results.
    pub results_dir: String,
    /// Persistent leaderboard file.
    pub leaderboard: String,
    /// Random hands per bot saved as hand replays (0 = off).
    pub replay_hands_per_bot: usize,
    /// Directory for hand replays (`<dir>/<run_id>/<bot>.txt`).
    pub replays_dir: String,
}

impl Default for RunSection {
    fn default() -> Self {
        RunSection {
            rounds: 1,
            seed: 0,
            threads: None,
            results_dir: "results".into(),
            leaderboard: "leaderboard.json".into(),
            replay_hands_per_bot: 0,
            replays_dir: "hand_replays".into(),
        }
    }
}

/// Game format section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FormatSection {
    /// Format name as registered in `engine::rules` (`"nlhe"`, `"plo"`, `"flhe"`, ...).
    pub variant: String,
    /// Seats per table (format maximum).
    #[serde(default = "default_seats")]
    pub max_seats: u8,
    /// Hole cards (default: 2 for Hold'em, 4 for PLO).
    pub hole_cards: Option<u8>,
    /// Fixed-limit small bet (default: the big blind).
    pub small_bet: Option<u64>,
    /// Fixed-limit big bet (default: twice the small bet).
    pub big_bet: Option<u64>,
    /// Fixed-limit cap on bets + raises per street (default 4).
    pub cap: Option<u8>,
}

fn default_seats() -> u8 {
    6
}

/// Table mode names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ModeName {
    /// Cash game.
    #[default]
    Cash,
    /// Tournament.
    Tournament,
}

/// Table section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableSection {
    /// Hands per table (maximum hands in tournaments).
    pub hands_per_table: u64,
    /// Starting stack in big blinds.
    #[serde(default = "default_stack")]
    pub starting_stack_bb: u64,
    /// Cash or tournament.
    #[serde(default)]
    pub mode: ModeName,
    /// Cash blinds.
    pub blinds: Option<Blinds>,
    /// Tournament blind levels.
    #[serde(default)]
    pub levels: Vec<BlindLevel>,
    /// Cash stack policy.
    #[serde(default)]
    pub stack_policy: StackPolicy,
    /// Duplicate mode.
    #[serde(default)]
    pub duplicate_mode: bool,
    /// Illegal action policy.
    #[serde(default)]
    pub illegal_action_policy: IllegalActionPolicy,
    /// Record decision times.
    #[serde(default)]
    pub record_timing: bool,
    /// Write text hand histories.
    #[serde(default)]
    pub hand_history: bool,
}

fn default_stack() -> u64 {
    100
}

/// One bot entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BotSection {
    /// Name (unique together with `version`).
    pub name: String,
    /// Optional version tag.
    pub version: Option<String>,
    /// Kind: a registered Rust bot or `python:module.ClassName`.
    pub kind: String,
    /// Arbitrary parameters passed to the factory.
    #[serde(default)]
    pub params: serde_json::Value,
    /// Fixed Elo rating: the bot is an anchor and its rating never changes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<f64>,
}

impl BotSection {
    /// For `python:module.ClassName` kinds, returns `module.ClassName`.
    pub fn python_target(&self) -> Option<&str> {
        self.kind.strip_prefix("python:")
    }
    /// Leaderboard key.
    pub fn key(&self) -> String {
        crate::runner::bot_key(&self.name, self.version.as_deref())
    }
}

/// A full config file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigFile {
    /// Run settings.
    #[serde(default)]
    pub run: RunSection,
    /// Game format.
    pub format: FormatSection,
    /// Table settings.
    pub table: TableSection,
    /// Matchmaking settings.
    #[serde(default)]
    pub matchmaking: MatchmakingConfig,
    /// Rating settings.
    #[serde(default)]
    pub rating: RatingConfig,
    /// Participating bots.
    pub bots: Vec<BotSection>,
}

impl ConfigFile {
    /// Parses TOML text.
    pub fn parse(text: &str) -> Result<ConfigFile, String> {
        toml::from_str(text).map_err(|e| e.to_string())
    }

    /// Reads and parses a file.
    pub fn load(path: &std::path::Path) -> Result<ConfigFile, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        Self::parse(&text)
    }

    /// Builds the game format: the registered format's defaults (see `engine::rules`), with
    /// the optional overrides applied and validated by that format's rules.
    pub fn game_format(&self) -> Result<GameFormat, String> {
        let f = &self.format;
        let rules = engine::rules::by_name(&f.variant).ok_or_else(|| {
            let known: Vec<&str> = engine::rules::ALL.iter().map(|r| r.name()).collect();
            format!(
                "unknown variant `{}` (known: {})",
                f.variant,
                known.join(", ")
            )
        })?;
        let mut fmt = rules.default_format(f.max_seats);
        if let Some(h) = f.hole_cards {
            fmt.hole_cards = h;
        }
        match &mut fmt.betting {
            Betting::FixedLimit {
                small_bet,
                big_bet,
                cap,
            } => {
                // Fixed-limit sizes default to the big blind and twice that.
                let bb = self.table_config()?.reference_bb();
                *small_bet = f.small_bet.unwrap_or(bb);
                *big_bet = f.big_bet.unwrap_or(2 * *small_bet);
                if let Some(c) = f.cap {
                    *cap = c;
                }
            }
            _ if f.small_bet.is_some() || f.big_bet.is_some() || f.cap.is_some() => {
                return Err("small_bet/big_bet/cap only apply to fixed-limit formats".into());
            }
            _ => {}
        }
        fmt.validate().map_err(|e| e.to_string())?;
        Ok(fmt)
    }

    /// Builds the table config.
    pub fn table_config(&self) -> Result<TableConfig, String> {
        let t = &self.table;
        let mode = match t.mode {
            ModeName::Cash => {
                if !t.levels.is_empty() {
                    return Err("`levels` is for tournaments; use `blinds` in cash mode".into());
                }
                TableMode::Cash {
                    blinds: t.blinds.ok_or("cash mode needs `blinds`")?,
                    stack_policy: t.stack_policy,
                }
            }
            ModeName::Tournament => {
                if t.blinds.is_some() {
                    return Err("tournaments use `levels`, not `blinds`".into());
                }
                TableMode::Tournament {
                    levels: t.levels.clone(),
                }
            }
        };
        let cfg = TableConfig {
            hands: t.hands_per_table,
            starting_stack_bb: t.starting_stack_bb,
            mode,
            duplicate: t.duplicate_mode,
            illegal_action_policy: t.illegal_action_policy,
            record_timing: t.record_timing,
            hand_history: t.hand_history,
        };
        cfg.validate().map_err(|e| e.to_string())?;
        Ok(cfg)
    }

    /// Builds the run spec.
    pub fn run_spec(&self) -> Result<RunSpec, String> {
        if self.bots.is_empty() {
            return Err("no [[bots]] configured".into());
        }
        let mut keys = std::collections::HashSet::new();
        for b in &self.bots {
            if !keys.insert(b.key()) {
                return Err(format!("duplicate bot `{}`", b.key()));
            }
        }
        Ok(RunSpec {
            format: self.game_format()?,
            table: self.table_config()?,
            matchmaking: self.matchmaking.clone(),
            rating: self.rating.clone(),
            rounds: self.run.rounds,
            seed: self.run.seed,
            threads: self.run.threads,
            replay_hands_per_bot: self.run.replay_hands_per_bot,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASIC: &str = r#"
        [run]
        rounds = 2
        seed = 7
        [format]
        variant = "nlhe"
        max_seats = 6
        [table]
        hands_per_table = 100
        blinds = { sb = 1, bb = 2 }
        [[bots]]
        name = "a"
        kind = "random"
        params = { fold = 1, raise_sizes = [0.5, 1.0] }
        [[bots]]
        name = "b"
        version = "2"
        kind = "python:examples.random_bot.RandomBot"
    "#;

    #[test]
    fn parse_basic() {
        let c = ConfigFile::parse(BASIC).unwrap();
        let spec = c.run_spec().unwrap();
        assert_eq!(spec.rounds, 2);
        assert_eq!(spec.table.starting_stack(), 200);
        assert_eq!(c.bots[0].params["raise_sizes"][1], 1.0);
        assert_eq!(c.bots[1].key(), "b@2");
        assert_eq!(
            c.bots[1].python_target(),
            Some("examples.random_bot.RandomBot")
        );
        assert_eq!(spec.matchmaking.min_appearances, 4);
    }

    #[test]
    fn rejects_unknown_fields_and_bad_modes() {
        assert!(ConfigFile::parse(&BASIC.replace("seed = 7", "seed = 7\nbogus = 1")).is_err());
        let c = ConfigFile::parse(&BASIC.replace("mode = \"cash\"", "")).unwrap();
        assert!(c.run_spec().is_ok());
        let t = BASIC.replace("blinds = { sb = 1, bb = 2 }", "mode = \"tournament\"");
        assert!(ConfigFile::parse(&t).unwrap().run_spec().is_err());
    }

    #[test]
    fn flhe_defaults() {
        let t = BASIC.replace("variant = \"nlhe\"", "variant = \"flhe\"");
        let f = ConfigFile::parse(&t).unwrap().game_format().unwrap();
        assert_eq!(
            f.betting,
            Betting::FixedLimit {
                small_bet: 2,
                big_bet: 4,
                cap: 4
            }
        );
    }
}
