# Condition Tables

Condition tables let you attach experimental metadata — such as drug treatments, concentrations, or replicate numbers — to specific wells in your plate. This information is stored in the OME-Zarr output and can be used by downstream analysis tasks.

## Format

A condition table is a **CSV file** with the following structure:

### Required Columns

| Column | Type | Description |
|---|---|---|
| `row` | string | Well row letter (A, B, C, ...) |
| `column` or `col` | integer | Well column number (1, 2, 3, ...) |

### Optional Columns

| Column | Type | Description |
|---|---|---|
| `acquisition` | integer | Acquisition ID. If present, conditions are filtered by acquisition. |
| *any other column* | string, int, float, or bool | Custom experimental metadata. |

!!! note "Column name matching"
    Column names are matched **case-insensitively**. Both `Row` and `row` work. For the column number, both `column` and `col` are accepted.

## Example

```csv
row,column,acquisition,drug,concentration,replicate
A,1,0,DMSO,0,1
A,2,0,DrugA,0.1,1
A,3,0,DrugA,1.0,1
B,1,0,DMSO,0,2
B,2,0,DrugA,0.1,2
B,3,0,DrugA,1.0,2
```

You can also have **multiple rows per well** to represent multiple conditions applied to the same well:

```csv
row,column,acquisition,drug,concentration,replicate
C,11,0,drugA,0.2,1
C,11,0,drugB,,2
```

## Data Type Rules

Each custom column must contain values of a **single type**. The converter auto-detects the type:

- **Strings**: Any text values. Empty strings, `NA`, `N/A`, and `Na` are treated as missing (`null`).
- **Numbers**: Integers or floats. `NaN` values are preserved.
- **Booleans**: `true` / `false` (case-sensitive).

Mixing types within a single column (e.g., some rows are strings, others are numbers) will cause an error.

## Usage

To use a condition table, set the `condition_table_path` in the `advanced` options of your acquisition:

!!! note
    The condition table path must be absolute.

## What Happens

When a condition table is provided:

1. For each well in the plate, the converter looks up matching rows in the CSV (by `row` + `column`, optionally filtered by `acquisition`).
2. The matched metadata columns are attached as **attributes** to the well's image tiles.
3. These attributes are written into:
   - At the OME-Zarr image level (each image contains the conditions that apply to it)
   - At the plate level (aggregated table of all image conditions)
   - Returned to the Fractal serve to be stored in datasets attributes
