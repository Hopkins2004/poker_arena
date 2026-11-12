//! Optional human-readable hand histories in a PokerStars-like text format, for debugging.
//!
//! Off by default. When enabled the table runner calls [`HandHistoryWriter::write_hand`] once
//! per finished hand. Hole cards of every player are printed ("Dealt to"), since this is a
//! debugging log for the operator, never shown to bots.

use engine::action::{ActionKind, Street};
use engine::card::Card;
use engine::eval::category;
use engine::{HandSetup, HandState, SeatStatus};
use std::fmt::Write;

/// Accumulates hand histories for one table.
#[derive(Debug, Default)]
pub struct HandHistoryWriter {
    out: String,
}

fn cards(cs: &[Card]) -> String {
    cs.iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

impl HandHistoryWriter {
    /// Empty writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the accumulated text.
    pub fn finish(self) -> String {
        self.out
    }

    /// Appends one completed hand.
    pub fn write_hand(&mut self, hand_id: u64, st: &HandState, setup: &HandSetup, names: &[&str]) {
        let o = &mut self.out;
        let fmt = st.format();
        let game = fmt.rules().title();
        let n = setup.n_seats;
        let _ = writeln!(
            o,
            "Hand #{hand_id}: {game} ({}/{}{}) - {}-max, Seat #{} is the button",
            setup.sb,
            setup.bb,
            if setup.ante > 0 {
                format!(" ante {}", setup.ante)
            } else {
                String::new()
            },
            fmt.max_seats,
            setup.button + 1
        );
        for s in 0..n {
            if setup.stacks[s] > 0 {
                let _ = writeln!(
                    o,
                    "Seat {}: {} ({} in chips)",
                    s + 1,
                    names[s],
                    setup.stacks[s]
                );
            }
        }
        let full_board: Vec<Card> = st.board().to_vec();
        let mut street = Street::Preflop;
        let mut hole_printed = false;
        for rec in st.history() {
            let who = names[rec.seat as usize];
            if !hole_printed
                && !matches!(
                    rec.kind,
                    ActionKind::PostAnte | ActionKind::PostSmallBlind | ActionKind::PostBigBlind
                )
            {
                write_hole(o, st, names, n);
                hole_printed = true;
            }
            while rec.street > street {
                street = street.next().unwrap();
                write_street(o, street, &full_board);
            }
            let allin = if rec.all_in { " and is all-in" } else { "" };
            let _ = match rec.kind {
                ActionKind::PostAnte => writeln!(o, "{who}: posts the ante {}{allin}", rec.amount),
                ActionKind::PostSmallBlind => {
                    writeln!(o, "{who}: posts small blind {}{allin}", rec.amount)
                }
                ActionKind::PostBigBlind => {
                    writeln!(o, "{who}: posts big blind {}{allin}", rec.amount)
                }
                ActionKind::Fold => writeln!(o, "{who}: folds"),
                ActionKind::Check => writeln!(o, "{who}: checks"),
                ActionKind::Call => writeln!(o, "{who}: calls {}{allin}", rec.amount),
                ActionKind::Bet => writeln!(o, "{who}: bets {}{allin}", rec.amount),
                ActionKind::Raise => writeln!(o, "{who}: raises to {}{allin}", rec.to),
            };
        }
        if !hole_printed {
            write_hole(o, st, names, n);
        }
        if let Some((seat, amount)) = st.uncalled_return() {
            let _ = writeln!(o, "Uncalled bet ({amount}) returned to {}", names[seat]);
        }
        // All-in runout: remaining streets dealt without action.
        while street.board_len() < full_board.len() {
            street = street.next().unwrap();
            write_street(o, street, &full_board);
        }
        if st.went_to_showdown() {
            let _ = writeln!(o, "*** SHOW DOWN ***");
            for s in 0..n {
                if let Some(h) = st.shown_cards(s) {
                    let _ = writeln!(
                        o,
                        "{}: shows [{}] ({:?})",
                        names[s],
                        cards(h),
                        category(st.strength(s))
                    );
                }
            }
        }
        for (i, pot) in st.pots().iter().enumerate() {
            let label = if i == 0 {
                "main pot".to_string()
            } else {
                format!("side pot-{i}")
            };
            for s in 0..n {
                if pot.winners & (1 << s) != 0 {
                    let _ = writeln!(o, "{} wins {label} ({} total)", names[s], pot.amount);
                }
            }
        }
        let _ = writeln!(o, "*** SUMMARY ***");
        if !full_board.is_empty() {
            let _ = writeln!(o, "Board [{}]", cards(&full_board));
        }
        for s in 0..n {
            if st.status(s) != SeatStatus::Empty {
                let _ = writeln!(
                    o,
                    "Seat {}: {} net {:+} (stack {})",
                    s + 1,
                    names[s],
                    st.net(s),
                    st.stack(s)
                );
            }
        }
        o.push('\n');
    }
}

fn write_hole(o: &mut String, st: &HandState, names: &[&str], n: usize) {
    let _ = writeln!(o, "*** HOLE CARDS ***");
    for s in 0..n {
        if st.status(s) != SeatStatus::Empty {
            let _ = writeln!(o, "Dealt to {} [{}]", names[s], cards(st.hole_cards(s)));
        }
    }
}

fn write_street(o: &mut String, street: Street, board: &[Card]) {
    let _ = match street {
        Street::Flop => writeln!(o, "*** FLOP *** [{}]", cards(&board[..3])),
        Street::Turn => writeln!(o, "*** TURN *** [{}] [{}]", cards(&board[..3]), board[3]),
        Street::River => writeln!(o, "*** RIVER *** [{}] [{}]", cards(&board[..4]), board[4]),
        Street::Preflop => Ok(()),
    };
}
