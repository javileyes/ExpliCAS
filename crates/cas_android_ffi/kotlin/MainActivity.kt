package es.javiergimenez.explicas

import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.Spinner
import android.widget.Switch
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

/**
 * Demo Activity for ExpliCAS engine integration.
 *
 * UI elements (add to activity_main.xml):
 * - EditText: id="editExpression"
 * - Button: id="btnEvaluate"
 * - TextView: id="txtResult"
 * - Spinner: id="spinnerPreset" (optional)
 * - Switch: id="switchStrict" (optional)
 * - TextView: id="txtAbiVersion" (optional)
 */
class MainActivity : AppCompatActivity() {

    private lateinit var editExpression: EditText
    private lateinit var btnEvaluate: Button
    private lateinit var txtResult: TextView
    private lateinit var spinnerPreset: Spinner
    private lateinit var switchStrict: Switch
    private lateinit var txtAbiVersion: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Bind views
        editExpression = findViewById(R.id.editExpression)
        btnEvaluate = findViewById(R.id.btnEvaluate)
        txtResult = findViewById(R.id.txtResult)
        spinnerPreset = findViewById(R.id.spinnerPreset)
        switchStrict = findViewById(R.id.switchStrict)
        txtAbiVersion = findViewById(R.id.txtAbiVersion)

        // Setup preset spinner
        val presets = arrayOf("cli", "small", "unlimited")
        spinnerPreset.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, presets)

        // Show ABI version
        txtAbiVersion.text = "Engine ABI: ${CasNative.abiVersion()}"

        // Evaluate button
        btnEvaluate.setOnClickListener {
            val expr = editExpression.text.toString()
            if (expr.isBlank()) {
                txtResult.text = "Enter an expression"
                return@setOnClickListener
            }
            evaluate(expr)
        }

        // Set default expression for testing
        editExpression.setText("x^2 + 2*x + 1")
    }

    private fun evaluate(expr: String) {
        val preset = spinnerPreset.selectedItem.toString()
        val mode = if (switchStrict.isChecked) "strict" else "best-effort"

        txtResult.text = "Evaluating..."
        btnEvaluate.isEnabled = false

        lifecycleScope.launch(Dispatchers.Default) {
            val result = try {
                evalAndExtract(expr, preset, mode)
            } catch (e: Exception) {
                "Exception: ${e.message}"
            }

            withContext(Dispatchers.Main) {
                txtResult.text = result
                btnEvaluate.isEnabled = true
            }
        }
    }

    /**
     * Call native library and extract result from JSON.
     */
    private fun evalAndExtract(expr: String, preset: String, mode: String): String {
        val optsJson = """{"budget":{"preset":"$preset","mode":"$mode"}}"""
        val json = CasNative.evalJson(expr, optsJson)

        val obj = JSONObject(json)
        val schemaVersion = obj.optInt("schema_version", 0)

        if (schemaVersion != 1) {
            return "Unexpected schema version: $schemaVersion"
        }

        val ok = obj.optBoolean("ok", true)

        return if (ok) {
            val result = obj.optString("result", "NO_RESULT")
            val timings = obj.optJSONObject("timings_us")
            val totalUs = timings?.optLong("total_us", 0) ?: 0
            "$result\n\n(${totalUs}µs)"
        } else {
            val error = obj.optJSONObject("error")
            val kind = error?.optString("kind", "Unknown") ?: "Unknown"
            val message = error?.optString("message", "No message") ?: "No message"
            val budget = obj.optJSONObject("budget")
            val budgetInfo = if (budget != null) {
                "\nBudget: ${budget.optString("preset")} / ${budget.optString("mode")}"
            } else ""
            "ERROR ($kind): $message$budgetInfo"
        }
    }
}
