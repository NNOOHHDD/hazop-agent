# Example: MCFC stack node

The stack node of a molten carbonate fuel cell (MCFC) system used for electric ship propulsion.

| File | Content |
|---|---|
| `pid.jpg` | P&ID of the node (metadata removed). |
| `node.md` | Node name, scope and design intent, written for this repository as a short summary (not the paper's text). |
| `reference_worksheet.csv` | 29-row reference HAZOP worksheet. |
| `results/` | Where the demo outputs go; see `results/README.md`. |

## Sources

Ahn J., Noh Y., Joung T., Lim Y., Kim J., Seo Y., Chang D. Safety integrity level (SIL) determination for a maritime fuel cell system as electric propulsion in accordance with IEC 61511. *International Journal of Hydrogen Energy* 44(5):3185-3194, 2019. doi:[10.1016/j.ijhydene.2018.12.065](https://doi.org/10.1016/j.ijhydene.2018.12.065)

- **Figure (`pid.jpg`)**: Reproduced from Ahn et al. (2019), © 2019 Hydrogen Energy Publications LLC, published by Elsevier, with permission of the corresponding author. Not covered by this repository's MIT License.
- **Table (`reference_worksheet.csv`)**: Restructured from Ahn et al. (2019).

## Notes on the reference worksheet

- Each row is one deviation -> cause -> consequence scenario of the published study. Causes, consequences and safeguards are condensed to key phrases; the published sentences are not copied.
- `guideword` and `parameter` give the deviation in canonical form (*Low* = *Less*, *High* = *More*).
- FI, SI and RI are the published values. The published table gives no SIL or recommendation per row, so those columns are empty.
- Row R08 (excess air) has no consequence or safeguard in the published table; it is kept as is.
